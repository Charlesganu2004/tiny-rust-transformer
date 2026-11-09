//! Lesson 1: read a training corpus and turn characters into token IDs.
//! Later pull requests add the Transformer and its training loop.

mod random;
mod autograd;

use std::collections::BTreeSet;

fn main() -> Result<(), String> {
    let text = std::fs::read_to_string("data/corpus.txt")
        .map_err(|error| error.to_string())?;

    // BTreeSet removes duplicate characters and keeps the vocabulary sorted.
    let vocabulary: Vec<char> = text.chars().collect::<BTreeSet<char>>()
        .into_iter()
        .collect();

    // Tokenization replaces each character with an index into the vocabulary.
    let tokens: Vec<usize> = text.chars()
        .map(|character| {
            vocabulary.iter().position(|&known| known == character).unwrap()
        })
        .collect();

    println!("Vocabulary ({} chars): {:?}", vocabulary.len(), vocabulary);
    println!("First 20 token IDs: {:?}", &tokens[..tokens.len().min(20)]);
    Ok(())
}
