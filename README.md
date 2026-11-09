# Tiny Rust Transformer: a language model you can read line by line

This is a **small, from-scratch autoregressive Transformer neural network written in Rust**. It learns from a short plain-text corpus and generates new text, one character at a time.

**It is a real trainable neural language model, but it is NOT a production-scale LLM or a pretrained chatbot.** It uses one causal attention head, one Transformer block, eight-dimensional embeddings, and a 12-character context. This is intentional: the entire implementation is inspectable. It does not have layer normalization, modern tokenization, GPU acceleration, a large dataset, or billions of parameters.

## Start here

**Read in this order:**

1. `src/main.rs` — reads text, makes a training example, and launches training.
2. `src/model.rs` (lines 9–76) — defines weights and model dimensions.
3. `src/model.rs` (lines 82–124) — performs matrix multiplication and softmax.
4. `src/model.rs` (lines 126–194) — runs the forward pass: embeddings, attention, MLP, logits.
5. `src/autograd.rs` — teaches the network derivatives / backpropagation.
6. `src/model.rs` (lines 197–228) — computes loss and updates weights.
7. `src/model.rs` (lines 230–284) — generates text and saves/loads a checkpoint.
8. `LINE_BY_LINE.md` — detailed explanations keyed to source line numbers.

## What the model does

```text
"rust " (a prompt, or part of training text)
        |
        v
Character IDs (one number per character)
        |
        v
Token embedding + positional embedding
        |
        v
Q/K/V projections
        |
        v
Causal scaled dot-product self-attention
        |
        v
Residual connection
        |
        v
2-layer MLP + ReLU + residual connection
        |
        v
Linear output head -> one score per possible next character
        |
        v
Softmax / sampling -> next character, then repeat
```

### During training

Each input is shifted against the target by one character:

```text
input:  r u s t   b u i l d s
next:   u s t   b u i l d s  
```

For the first character `r`, it should predict `u`; for `u`, it should predict `s`, and so on. Training uses *cross entropy* to measure error and *Adam* to update weights. The calculation graph and the derivatives are implemented here, with **no external libraries**.

## Requirements

Rust and Cargo (the standard Rust toolchain). Install from https://rustup.rs/ if necessary.

In a terminal, open the project folder:

```bash
cd tiny-rust-transformer
cargo test
cargo run --release -- train 500
cargo run --release -- generate "rust " 100 0.8
cargo run --release -- inspect
```

`train 500` means 500 weight updates; the training command saves `model.txt`. Running `generate` requires having trained first. The last `0.8` controls sampling temperature: smaller values are less random; larger values are more random.

**Note:** If you change `data/corpus.txt`, retrain before generating. Prompts can contain only characters that occur in the training corpus. The tiny dataset and short context limit the quality; expect imperfect text.

### Windows PowerShell

```powershell
cd .\tiny-rust-transformer
cargo test
cargo run --release -- train 500
cargo run --release -- generate "rust " 100 0.8
```

## What's in each file?

| File | Purpose |
|---|---|
| `Cargo.toml` | Rust project metadata; zero dependencies. |
| `data/corpus.txt` | Sample training text; replace it with your own. |
| `src/main.rs` | Command-line interface, input encoding, training loop. |
| `src/model.rs` | Causal self-attention, MLP, logits, loss, Adam, text generation, model checkpoints. |
| `src/autograd.rs` | A scalar computation graph and chain-rule gradients. |
| `src/random.rs` | Tiny seeded pseudorandom number generator. |
| `LINE_BY_LINE.md` | Guided reading with exact code line references. |
| `EXERCISES.md` | Small edits you can make while learning. |

## Important technical boundaries

- **Neural architecture:** 1 Transformer-style decoder block, 1 attention head; char-level next-token prediction.
- **Training:** the input text is the only source of knowledge. It does not import GPT weights or use an API.
- **Compute:** CPU-only; the intentionally simple scalar autograd may be slow for large training loops.
- **Scale:** this implementation is an educational *mini language model*, not a useful general-purpose LLM.
- **Causal mask:** loop `for j in 0..=t` makes it impossible for the prediction at time `t` to access future characters.
- **Security:** `random.rs` is not a cryptographic random number generator. Checkpoint files and corpus text should be treated as untrusted if coming from other people; no robust format hardening is implemented.

## Debug it line by line

Open `src/main.rs` in VS Code, click next to **line 30** to set a breakpoint, and run in debug mode. Continue to line 41 to step into `Model::train_one`; then step into `Model::forward` and inspect `x`, `q`, `k`, `v`, `scores`, `attention`, and `logits`. Starting with `cargo run -- train 1` makes it easier to inspect one update.

Start with **LINE_BY_LINE.md** and move to `src/autograd.rs` when you want to understand how the weights learn.
## GitHub PR learning path

This repository was constructed as **eight stacked draft pull requests**. Each PR targets the preceding learning branch, not `main`, so its **Files changed** tab shows only that lesson:

1. [Lesson 1 — Corpus and tokenizer](https://github.com/Charlesganu2004/tiny-rust-transformer/pull/1)
2. [Lesson 2 — Autograd](https://github.com/Charlesganu2004/tiny-rust-transformer/pull/2)
3. [Lesson 3 — Model weights and linear layers](https://github.com/Charlesganu2004/tiny-rust-transformer/pull/3)
4. [Lesson 4 — Causal self-attention](https://github.com/Charlesganu2004/tiny-rust-transformer/pull/4)
5. [Lesson 5 — Loss and optimizer](https://github.com/Charlesganu2004/tiny-rust-transformer/pull/5)
6. [Lesson 6 — Generation and checkpoints](https://github.com/Charlesganu2004/tiny-rust-transformer/pull/6)
7. [Lesson 7 — Runnable CLI](https://github.com/Charlesganu2004/tiny-rust-transformer/pull/7)
8. [Lesson 8 — Tests and study guide](https://github.com/Charlesganu2004/tiny-rust-transformer/pull/8)

**How to participate:** Open one PR, choose `Files changed`, highlight individual lines with the blue **+**, and add your own explanation/questions as inline comments. Submit your review and optionally push your own code changes. Later, ask me to review that PR. I'll inspect its current diff, review threads, and commits before suggesting or making fixes. **Nothing will be merged automatically.**

To read the integrated project in one place before the stack is merged, switch GitHub's branch selector to `learn/08-tests-and-study-guide` once that branch is available.
