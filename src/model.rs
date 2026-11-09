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
}
