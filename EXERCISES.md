# Exercises: turn the code into a series of Rust + ML lessons

## Exercise 1 — Change the training corpus

Edit `data/corpus.txt` to include many repetitions of your own short sentences (use lowercase and ordinary punctuation at first). Then run `cargo run --release -- train 500`. What patterns does the model reproduce? What happens with only one or two examples?

## Exercise 2 — Change embedding size

Edit `D` in `src/model.rs` from `8` to `12`. This gives each character more features but creates more weights and operations. Train again and compare the loss.

## Exercise 3 — Change the context window

Edit `CONTEXT` from `12` to `16` and retrain. Longer context means the model can look farther back, but attention performs more work. Notice how `for j in 0..=t` still blocks future positions.

## Exercise 4 — Study temperature

After training, run:

```bash
cargo run --release -- generate "rust " 100 0.3
cargo run --release -- generate "rust " 100 0.8
cargo run --release -- generate "rust " 100 1.4
```

Predict which output will be more repetitive and which will be more random. The random seed is fixed for reproducibility, but the distribution changes with temperature.

## Exercise 5 — Find the gradients

Add this temporary line inside `Model::train_one`, right after `let gradients...`:

```rust
println!("first weight gradient = {}", gradients[0]);
```

Step through `Tape::backward` to see where that number comes from. Then remove the debug print.

## Exercise 6 — Explain each layer

In `src/model.rs`, explain in your own words: `embeddings`, `positions`, `wq`, `wk`, `wv`, `wo`, `w1`, `w2`, and `head`. Each has different dimensions and a different job.

## Exercise 7 — Understand a missing feature

This code deliberately omits layer normalization, multi-head attention, and stacked blocks. Look at a modern decoder-only Transformer architecture and try to identify where these would be inserted. Adding them would require both a forward calculation and correct autograd derivatives.

## Exercise 8 — Add a simple validation set

Right now training loss is measured on sampled training text. Add a second text file, calculate its loss without calling `Adam::update`, and track whether the model generalizes or memorizes the training data.