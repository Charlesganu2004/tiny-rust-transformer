//! Run with: cargo run --release -- train 500
//! Then:     cargo run --release -- generate "rust " 100
//! Or:       cargo run --release -- inspect

mod autograd;  // Our tiny automatic differentiation engine.
mod model;     // Transformer architecture, training, sampling, and checkpointing.
mod random;    // Dependency-free reproducible random numbers.

use crate::model::{Adam, Model, CONTEXT};
use crate::random::Random;
use std::collections::BTreeSet;
use std::env;

const CORPUS_PATH: &str = "data/corpus.txt";
const CHECKPOINT_PATH: &str = "model.txt";

fn train(steps: usize) -> Result<(), String> {
    // Read a tiny training corpus; no internet or pretrained model is used.
    let corpus = std::fs::read_to_string(CORPUS_PATH).map_err(|e| e.to_string())?;
    // Collect all unique characters in stable, sorted order.
    let vocab: Vec<char> = corpus.chars().collect::<BTreeSet<char>>().into_iter().collect();
    let encoded: Vec<usize> = corpus.chars()
        .map(|ch| vocab.iter().position(|&candidate| candidate == ch).unwrap())
        .collect();
    if encoded.len() < CONTEXT + 2 {
        return Err(format!("Corpus needs at least {} characters", CONTEXT + 2));
    }

    // Create a random transformer and Adam's gradient-history buffers.
    let mut model = Model::new(vocab, 42);
    let mut adam = Adam::new(model.parameter_count());
    let mut rng = Random::new(2026);
    println!("Characters: {} | Parameters: {}", model.vocab.len(), model.parameter_count());

    for step in 1..=steps {
        // Pick a random CONTEXT+1-character slice from the corpus.
        let start = (rng.next_u32() as usize) % (encoded.len() - CONTEXT);
        let window = &encoded[start..start + CONTEXT + 1];
        let input = &window[..CONTEXT]; // Characters the network sees.
        let target = &window[1..];      // Each character shifted forward by one.
        let loss = model.train_one(input, target, &mut adam, 0.005);
        if step == 1 || step % 50 == 0 || step == steps {
            println!("step {:>5}/{steps}: cross-entropy loss = {:.4}", step, loss);
        }
    }

    // Save the trained weights and vocabulary for a separate generation command.
    model.save(CHECKPOINT_PATH)?;
    println!("Saved {CHECKPOINT_PATH}");
    println!("Sample:\n{}", model.generate("rust ", 120, 0.8, 7)?);
    Ok(())
}

fn main() -> Result<(), String> {
    let arguments: Vec<String> = env::args().collect();
    match arguments.get(1).map(String::as_str) {
        Some("train") => {
            let steps: usize = arguments.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);
            train(steps)
        }
        Some("generate") => {
            let model = Model::load(CHECKPOINT_PATH)?;
            let prompt = arguments.get(2).map(String::as_str).unwrap_or("rust ");
            let length = arguments.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
            let temperature = arguments.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.8);
            println!("{}", model.generate(prompt, length, temperature, 99)?);
            Ok(())
        }
        Some("inspect") => {
            let model = Model::load(CHECKPOINT_PATH)?;
            println!("Vocabulary: {:?}", model.vocab);
            println!("Parameter count: {}", model.parameter_count());
            println!("Max context: {CONTEXT} characters");
            Ok(())
        }
        _ => {
            println!("Commands:\n  cargo run --release -- train 500\n  cargo run --release -- generate \"rust \" 100 0.8\n  cargo run --release -- inspect\n  cargo test");
            Ok(())
        }
    }
}