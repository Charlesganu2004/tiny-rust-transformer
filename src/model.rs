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
}
