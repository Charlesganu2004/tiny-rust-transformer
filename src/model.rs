//! A very small, character-level, decoder-only Transformer.
//! Pipeline: character IDs -> embeddings -> masked self-attention -> MLP -> logits.
//! We deliberately omit layer normalization and multiple layers to keep it readable.

use crate::autograd::{Tape, Var};
use crate::random::Random;
use std::ops::Range;

pub const CONTEXT: usize = 12; // The model sees at most 12 previous characters.
pub const D: usize = 8;        // Each token is represented by 8 numbers.
const HIDDEN: usize = 16;      // Width of the feed-forward neural network.

// Each range identifies a matrix or bias vector inside one flat weights Vec.
struct Layout {
    embeddings: Range<usize>,   // [vocab_size, D]
    positions: Range<usize>,    // [CONTEXT, D]
    wq: Range<usize>,           // [D, D] query projection.
    wk: Range<usize>,           // [D, D] key projection.
    wv: Range<usize>,           // [D, D] value projection.
    wo: Range<usize>,           // [D, D] attention output projection.
    w1: Range<usize>,           // [D, HIDDEN] first MLP layer.
    b1: Range<usize>,           // [HIDDEN] first MLP bias.
    w2: Range<usize>,           // [HIDDEN, D] second MLP layer.
    b2: Range<usize>,           // [D] second MLP bias.
    head: Range<usize>,         // [D, vocab_size] prediction layer.
    head_bias: Range<usize>,    // [vocab_size] prediction bias.
    total: usize,
}

// Give each trainable matrix a contiguous section of the weights array.
fn allocate(cursor: &mut usize, count: usize) -> Range<usize> {
    let start = *cursor;
    *cursor += count;
    start..*cursor
}

impl Layout {
    fn new(vocab_size: usize) -> Self {
        let mut c = 0;
        let embeddings = allocate(&mut c, vocab_size * D);
        let positions = allocate(&mut c, CONTEXT * D);
        let wq = allocate(&mut c, D * D);
        let wk = allocate(&mut c, D * D);
        let wv = allocate(&mut c, D * D);
        let wo = allocate(&mut c, D * D);
        let w1 = allocate(&mut c, D * HIDDEN);
        let b1 = allocate(&mut c, HIDDEN);
        let w2 = allocate(&mut c, HIDDEN * D);
        let b2 = allocate(&mut c, D);
        let head = allocate(&mut c, D * vocab_size);
        let head_bias = allocate(&mut c, vocab_size);
        Self { embeddings, positions, wq, wk, wv, wo, w1, b1, w2, b2, head, head_bias, total: c }
    }
}

pub struct Model {
    pub vocab: Vec<char>,      // Index-to-character lookup table.
    weights: Vec<f32>,        // Everything that training can change.
    layout: Layout,
}

impl Model {
    pub fn new(vocab: Vec<char>, seed: u64) -> Self {
        let layout = Layout::new(vocab.len());
        let mut rng = Random::new(seed);
        let mut weights: Vec<f32> = (0..layout.total)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * 0.10)
            .collect();
        // Start biases at zero; biases still learn during training.
        for i in layout.b1.clone().chain(layout.b2.clone()).chain(layout.head_bias.clone()) {
            weights[i] = 0.0;
        }
        Self { vocab, weights, layout }
    }

    pub fn parameter_count(&self) -> usize {
        self.weights.len()
    }

    // Matrix multiplication for a single row: output = input * W + bias.
    // We keep explicit loops so you can see each multiply and add.
    fn linear(
        tape: &mut Tape,
        params: &[Var],
        input: &[Var],
        matrix: &Range<usize>,
        output_size: usize,
        bias: Option<&Range<usize>>,
    ) -> Vec<Var> {
        let mut output = Vec::with_capacity(output_size);
        for col in 0..output_size {
            let mut total = match bias {
                Some(b) => params[b.start + col],
                None => tape.leaf(0.0),
            };
            for row in 0..input.len() {
                let weight = params[matrix.start + row * output_size + col];
                let product = tape.mul(input[row], weight);
                total = tape.add(total, product);
            }
            output.push(total);
        }
        output
    }

    fn add_vectors(tape: &mut Tape, a: &[Var], b: &[Var]) -> Vec<Var> {
        a.iter().zip(b).map(|(&x, &y)| tape.add(x, y)).collect()
    }

    // Softmax converts attention scores into nonnegative weights that sum to 1.
    // Subtracting a detached maximum prevents overflow without changing results.
    fn softmax(tape: &mut Tape, scores: &[Var]) -> Vec<Var> {
        let max = scores.iter().map(|&x| tape.value(x)).fold(f32::NEG_INFINITY, f32::max);
        let max_constant = tape.leaf(max);
        let mut exponentials = Vec::with_capacity(scores.len());
        let mut denominator = tape.leaf(0.0);
        for &score in scores {
            let shifted = tape.sub(score, max_constant);
            let e = tape.exp(shifted);
            denominator = tape.add(denominator, e);
            exponentials.push(e);
        }
        exponentials.into_iter().map(|e| tape.div(e, denominator)).collect()
    }

    // Forward pass: compute predictions for every position in an input window.
    // CAUSALITY: position t can only attend to positions j <= t.
    pub fn forward(&self, tape: &mut Tape, params: &[Var], tokens: &[usize]) -> Vec<Vec<Var>> {
        assert!(!tokens.is_empty() && tokens.len() <= CONTEXT);
        let time_steps = tokens.len();

        // Embedding: look up each token, and add where it occurs in the window.
        let mut x = Vec::with_capacity(time_steps);
        for (position, &token) in tokens.iter().enumerate() {
            let mut token_vector = Vec::with_capacity(D);
            for dimension in 0..D {
                let e = params[self.layout.embeddings.start + token * D + dimension];
                let p = params[self.layout.positions.start + position * D + dimension];
                token_vector.push(tape.add(e, p));
            }
            x.push(token_vector);
        }

        // Q = X Wq, K = X Wk, V = X Wv.
        let q: Vec<Vec<Var>> = x.iter()
            .map(|row| Self::linear(tape, params, row, &self.layout.wq, D, None)).collect();
        let k: Vec<Vec<Var>> = x.iter()
            .map(|row| Self::linear(tape, params, row, &self.layout.wk, D, None)).collect();
        let v: Vec<Vec<Var>> = x.iter()
            .map(|row| Self::linear(tape, params, row, &self.layout.wv, D, None)).collect();

        let mut all_logits = Vec::with_capacity(time_steps);
        for t in 0..time_steps {
            let mut scores = Vec::with_capacity(t + 1);
            // Attention score(t,j) = dot(Q[t], K[j]) / sqrt(D).
            for j in 0..=t { // Causal mask: future tokens are NOT examined.
                let mut dot_product = tape.leaf(0.0);
                for dimension in 0..D {
                    let product = tape.mul(q[t][dimension], k[j][dimension]);
                    dot_product = tape.add(dot_product, product);
                }
                scores.push(tape.scale(dot_product, 1.0 / (D as f32).sqrt()));
            }
            let attention = Self::softmax(tape, &scores);

            // Context[t] = sum_j(attention[t,j] * V[j]).
            let mut context = vec![tape.leaf(0.0); D];
            for j in 0..=t {
                for dimension in 0..D {
                    let contribution = tape.mul(attention[j], v[j][dimension]);
                    context[dimension] = tape.add(context[dimension], contribution);
                }
            }

            // Attention output and first residual connection.
            let projected = Self::linear(tape, params, &context, &self.layout.wo, D, None);
            let h = Self::add_vectors(tape, &x[t], &projected);

            // Two-layer feed-forward neural net: ReLU(h*W1+b1)*W2+b2.
            let mut hidden = Self::linear(tape, params, &h, &self.layout.w1, HIDDEN, Some(&self.layout.b1));
            for value in &mut hidden {
                *value = tape.relu(*value);
            }
            let mlp = Self::linear(tape, params, &hidden, &self.layout.w2, D, Some(&self.layout.b2));
            let output = Self::add_vectors(tape, &h, &mlp); // Second residual connection.

            // One logit (unnormalized score) for each possible next character.
            let logits = Self::linear(
                tape, params, &output, &self.layout.head, self.vocab.len(), Some(&self.layout.head_bias),
            );
            all_logits.push(logits);
        }
        all_logits
    }

    // Cross entropy = log(sum(exp(logits))) - logit[correct_character].
    fn cross_entropy(tape: &mut Tape, logits: &[Var], target: usize) -> Var {
        let max = logits.iter().map(|&x| tape.value(x)).fold(f32::NEG_INFINITY, f32::max);
        let shift = tape.leaf(max); // Stability; treated as a constant for gradients.
        let mut sum = tape.leaf(0.0);
        for &logit in logits {
            let shifted = tape.sub(logit, shift);
            let exp = tape.exp(shifted);
            sum = tape.add(sum, exp);
        }
        let log_sum = tape.ln(sum);
        let log_normalizer = tape.add(log_sum, shift);
        tape.sub(log_normalizer, logits[target])
    }

    // One complete update: forward -> loss -> backward -> optimizer step.
    pub fn train_one(&mut self, inputs: &[usize], targets: &[usize], adam: &mut Adam, rate: f32) -> f32 {
        assert_eq!(inputs.len(), targets.len());
        let mut tape = Tape::new();
        let parameters: Vec<Var> = self.weights.iter().map(|&w| tape.leaf(w)).collect();
        let logits = self.forward(&mut tape, &parameters, inputs);
        let mut loss = tape.leaf(0.0);
        for (row, &correct) in logits.iter().zip(targets) {
            let penalty = Self::cross_entropy(&mut tape, row, correct);
            loss = tape.add(loss, penalty);
        }
        let loss = tape.scale(loss, 1.0 / inputs.len() as f32);
        let loss_value = tape.value(loss);
        tape.backward(loss);
        let gradients: Vec<f32> = parameters.iter().map(|&p| tape.grad(p)).collect();
        adam.update(&mut self.weights, &gradients, rate);
        loss_value
    }

    // Generate one character at a time from the most recent CONTEXT characters.
    pub fn generate(&self, prompt: &str, length: usize, temperature: f32, seed: u64) -> Result<String, String> {
        if prompt.is_empty() { return Err("Prompt cannot be empty".to_string()); }
        if temperature <= 0.0 { return Err("Temperature must be positive".to_string()); }
        let mut ids = Vec::new();
        for ch in prompt.chars() {
            let id = self.vocab.iter().position(|&known| known == ch)
                .ok_or_else(|| format!("Unknown character {:?}; edit the corpus and retrain", ch))?;
            ids.push(id);
        }
        let mut text = prompt.to_string();
        let mut rng = Random::new(seed);
        for _ in 0..length {
            let start = ids.len().saturating_sub(CONTEXT);
            let window = &ids[start..];
            let mut tape = Tape::new();
            let parameters: Vec<Var> = self.weights.iter().map(|&w| tape.leaf(w)).collect();
            let logits = self.forward(&mut tape, &parameters, window);
            let last = logits.last().expect("Nonempty prompt");
            let scores: Vec<f32> = last.iter().map(|&v| tape.value(v) / temperature).collect();
            let next = sample_index(&scores, &mut rng);
            ids.push(next);
            text.push(self.vocab[next]);
        }
        Ok(text)
    }

    // Simple human-readable checkpoint: header, Unicode characters, and weights.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let mut text = format!("TINY_RUST_TRANSFORMER_V1\n{}\n", self.vocab.len());
        for &ch in &self.vocab { text.push_str(&format!("{} ", ch as u32)); }
        text.push('\n');
        text.push_str(&format!("{}\n", self.weights.len()));
        for &w in &self.weights { text.push_str(&format!("{w:.9}\n")); }
        std::fs::write(path, text).map_err(|error| error.to_string())
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
        let mut fields = text.split_whitespace();
        if fields.next() != Some("TINY_RUST_TRANSFORMER_V1") {
            return Err("Not a Tiny Rust Transformer checkpoint".to_string());
        }
        let n: usize = parse_next(&mut fields)?;
        let mut vocab = Vec::with_capacity(n);
        for _ in 0..n {
            let code: u32 = parse_next(&mut fields)?;
            vocab.push(char::from_u32(code).ok_or("Invalid Unicode character")?);
        }
        let count: usize = parse_next(&mut fields)?;
        let mut model = Self::new(vocab, 1); // Builds the same parameter layout.
        if count != model.weights.len() { return Err("Checkpoint shape mismatch".to_string()); }
        for w in &mut model.weights { *w = parse_next(&mut fields)?; }
        Ok(model)
    }
}

// Helper for converting space-separated checkpoint fields into Rust numbers.
fn parse_next<T: std::str::FromStr>(parts: &mut std::str::SplitWhitespace<'_>) -> Result<T, String>
where T::Err: std::fmt::Display {
    let raw = parts.next().ok_or_else(|| "Checkpoint ended early".to_string())?;
    raw.parse::<T>().map_err(|error| error.to_string())
}

// Convert logits to probabilities, then sample using a random number.
fn sample_index(logits: &[f32], rng: &mut Random) -> usize {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let mut threshold = rng.next_f32() * sum;
    for (index, probability) in exp.iter().enumerate() {
        threshold -= probability;
        if threshold <= 0.0 { return index; }
    }
    logits.len() - 1
}

// Adam optimizer keeps an exponential average of gradients and squared gradients.
pub struct Adam {
    first: Vec<f32>,
    second: Vec<f32>,
    step: i32,
}

impl Adam {
    pub fn new(count: usize) -> Self {
        Self { first: vec![0.0; count], second: vec![0.0; count], step: 0 }
    }

    pub fn update(&mut self, weights: &mut [f32], gradients: &[f32], learning_rate: f32) {
        self.step += 1;
        let norm = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();
        let clip = if norm > 1.0 { 1.0 / norm } else { 1.0 };
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let correction1 = 1.0 - beta1.powi(self.step);
        let correction2 = 1.0 - beta2.powi(self.step);
        for i in 0..weights.len() {
            let g = gradients[i] * clip;
            self.first[i] = beta1 * self.first[i] + (1.0 - beta1) * g;
            self.second[i] = beta2 * self.second[i] + (1.0 - beta2) * g * g;
            let mean = self.first[i] / correction1;
            let variance = self.second[i] / correction2;
            weights[i] -= learning_rate * mean / (variance.sqrt() + 1e-8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_mask_prevents_future_leakage() {
        let model = Model::new(vec!['a', 'b', 'c'], 10);
        let mut tape_a = Tape::new();
        let params_a: Vec<Var> = model.weights.iter().map(|&w| tape_a.leaf(w)).collect();
        let left = model.forward(&mut tape_a, &params_a, &[0, 1, 0]);
        let first_a: Vec<f32> = left[0].iter().map(|&v| tape_a.value(v)).collect();
        let mut tape_b = Tape::new();
        let params_b: Vec<Var> = model.weights.iter().map(|&w| tape_b.leaf(w)).collect();
        let right = model.forward(&mut tape_b, &params_b, &[0, 2, 2]);
        let first_b: Vec<f32> = right[0].iter().map(|&v| tape_b.value(v)).collect();
        assert_eq!(first_a, first_b); // Changing future tokens cannot affect position 0.
    }

    #[test]
    fn checkpoint_roundtrip() {
        let model = Model::new(vec!['a', 'b', ' '], 23);
        let path = std::env::temp_dir().join(format!("tiny-rust-transformer-test-{}.txt", std::process::id()));
        model.save(path.to_str().unwrap()).unwrap();
        let loaded = Model::load(path.to_str().unwrap()).unwrap();
        assert_eq!(model.vocab, loaded.vocab);
        for (original, restored) in model.weights.iter().zip(&loaded.weights) {
            assert!((original - restored).abs() < 1e-8);
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn training_reduces_loss_on_repeated_example() {
        let mut model = Model::new(vec!['a', 'b'], 5);
        let mut adam = Adam::new(model.parameter_count());
        let inputs = [0, 1, 0, 1];
        let targets = [1, 0, 1, 0];
        let start = model.train_one(&inputs, &targets, &mut adam, 0.005);
        let mut end = start;
        for _ in 0..50 { end = model.train_one(&inputs, &targets, &mut adam, 0.005); }
        assert!(end < start, "Loss failed to decrease: {start} -> {end}");
    }
}