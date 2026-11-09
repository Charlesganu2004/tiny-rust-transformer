# Line-by-line code walkthrough

The line numbers below refer to the unmodified `.rs` files in this project. A code editor such as VS Code displays these numbers automatically. Read this guide alongside the source.

## Lesson 1 — `src/main.rs`: how the program trains

| Line(s) | What the code means |
|---|---|
| 5 | `mod autograd` makes the local `autograd.rs` module available. |
| 6 | `mod model` adds the Transformer implementation. |
| 7 | `mod random` adds the sample random-number generator. |
| 9 | `use` imports three names from `model` so we can write `Model`, not `model::Model`. |
| 10 | Import the `Random` type. |
| 11 | `BTreeSet` stores characters in sorted order without duplicates. |
| 12 | `env` lets Rust read command-line arguments. |
| 14 | The text file used to train the language model. |
| 15 | File in which we save the trained neural-network weights. |
| 17 | `fn train(steps: usize)` defines a function. `steps` is an unsigned integer. `Result<(), String>` means success or a textual error. |
| 19 | Read the text file. `map_err` converts a filesystem error to text. `?` returns early if reading fails. |
| 21 | Extract unique sorted characters, e.g., `[' ', 'a', 'b', ...]`. This is the vocabulary. |
| 22–24 | Convert each character to its numeric index in the vocabulary. `.map` transforms items; `.collect` makes a `Vec<usize>`. |
| 25–27 | Reject a corpus too short to supply one complete input/target window. |
| 30 | Construct the model with random initial weights. `mut` allows us to change it while training. `42` is a fixed seed. |
| 31 | Make Adam's two running-statistic arrays, with one entry per trainable weight. |
| 32 | Make another reproducible random-number stream for picking training samples. |
| 33 | Print the number of characters and trainable parameters. |
| 35 | Repeat from step 1 through the specified number of updates (inclusive Rust range `..=`). |
| 37 | Choose a starting position in the corpus. `%` is remainder, which keeps the position in bounds. |
| 38 | Slice a section of `CONTEXT + 1` numeric character IDs, without copying data. |
| 39 | `&window[..CONTEXT]` is the first 12 characters of the window. |
| 40 | `&window[1..]` is the same text shifted forward one character. These are the correct next-character answers. |
| 41 | Train on this window, compute the error, backpropagate, and update the weights. `0.005` is the learning rate. |
| 42–44 | Print a loss value on the first step, every 50 steps, and the last step. |
| 48 | Write the trained model to `model.txt`. |
| 50 | Generate 120 new characters beginning with the seed string `"rust "`. |
| 51 | `Ok(())` means the function succeeded with no return payload. |
| 54 | The real entry point of the executable. |
| 55 | Read terminal arguments, including `train`, `500`, `generate`, and prompts. |
| 56 | `match` chooses different commands based on the first user argument. |
| 57–60 | Train mode. If steps are missing or invalid, use 500. |
| 61–68 | Generation mode. Load saved model, read optional prompt/length/temperature, and print text. |
| 69–75 | Inspect mode shows the vocabulary, trainable parameter count, and maximum context. |
| 76–79 | For no/unknown command, print usage instructions. |

### Worked example: Why shift the target by one?

Suppose the training text is `"rust"`. If a context window includes `r`, `u`, and `s`, the network sees:

```text
Position       0   1   2
Input          r   u   s
Correct next   u   s   t
```

It calculates one prediction per position. Its loss is lower when it gives more probability to the correct next character. It does not simply memorize a label for the entire sentence; it gets feedback at every position.

## Lesson 2 — `src/model.rs`: what the variables mean

| Line(s) | Explanation |
|---|---|
| 9 | `CONTEXT = 12`: one forward pass can contain 1–12 characters. |
| 10 | `D = 8`: represent each character with eight numbers. |
| 11 | `HIDDEN = 16`: width of the MLP hidden layer. |
| 14–28 | `Layout` describes where each matrix lives inside one big array of `f32` weights. |
| 31–35 | `allocate` reserves a consecutive range of array indices and advances the counter. |
| 38–52 | Allocate token embeddings, positional embeddings, Q/K/V matrices, attention output weights, MLP weights, and the prediction head. |
| 56–60 | `Model` owns the vocabulary, trainable weights, and matrix layout. |
| 63–74 | Initialize weights with small pseudorandom values, and initialize biases to zero. |
| 76–78 | Number of trainable scalar parameters. |

### What is a matrix weight?

Suppose `input = [2, 3]` and a matrix is:

```text
W = [ [1, 4],
      [5, 6] ]
```

Then `input * W` is `[2*1 + 3*5, 2*4 + 3*6]` = `[17, 26]`. In code, this is exactly the nested loop in `linear`.

| Line(s) | Explanation |
|---|---|
| 82–89 | `linear` accepts a computation graph, its weight variables, an input vector, a weight matrix, an output width, and optionally a bias vector. |
| 90 | Reserve space for the output values. |
| 91 | Repeat for every output dimension. |
| 92–95 | Start the sum from a trainable bias, or zero if there is no bias. |
| 96 | Visit each element of the input vector. |
| 97 | Look up the correct weight in the flattened row-major matrix. |
| 98 | Multiply the input number by the weight, adding that operation to the autograd graph. |
| 99 | Add the product to the running sum. |
| 101–104 | Store all output values and return them. |
| 106–108 | Element-wise vector addition. This implements residual connections later. |
| 112–124 | Stable softmax: subtract the largest score, exponentiate, divide by their sum. |

## Lesson 3 — `src/model.rs`: the attention forward pass

| Line(s) | Explanation |
|---|---|
| 128–130 | Verify there is at least one character, and no more than the model's context capacity. |
| 133–142 | Construct one input vector per character by adding token and position embeddings. |
| 137 | Choose the trainable embedding for this specific character and embedding dimension. |
| 138 | Choose the trainable positional embedding for its location in the window. |
| 139 | Add the two together. |
| 145–150 | Multiply the input vectors by separate matrices to make **Q (queries)**, **K (keys)**, and **V (values)**. |
| 153 | For each input position `t`, compute one output/prediction. |
| 156 | **Causal mask:** `0..=t` means only this character and earlier characters are visible. No future token can leak into the result. |
| 157–163 | Find similarity between `Q[t]` and each permitted `K[j]` using their dot product, scaled by `sqrt(D)`. |
| 164 | Use softmax so attention weights are positive and total 1. |
| 167–173 | Multiply each `V[j]` vector by its attention weight and add results. This creates the context vector. |
| 176 | Send context through a trainable output projection. |
| 177 | Add the original embedding vector to the attention output (residual shortcut). |
| 180 | Send the result through the first MLP layer: eight inputs become 16 hidden values. |
| 181–183 | Apply ReLU: negative values become zero; positive values stay positive. |
| 184 | Second MLP layer converts 16 values back to eight. |
| 185 | Add the MLP output to the earlier vector (another residual shortcut). |
| 188–190 | The output head produces a score for every character in the vocabulary. |
| 191–194 | Store predictions at each position and return them. |

### Why are attention scores useful?

At the last character of `"rust "`, the model builds a query for that position. It compares it against keys for `r`, `u`, `s`, `t`, and space. It turns those scores into attention weights and uses the weighted value vectors to build a context-aware representation. A trained model may learn to put more weight on characters that help it predict the next one.

**Q**, **K**, and **V** are learned projections of the embeddings; they are not words or lookup tables of facts. All weights start random and change based on the next-character training objective.

## Lesson 4 — `src/autograd.rs`: why the model learns

| Line(s) | Explanation |
|---|---|
| 6 | `Var(pub usize)` is a pointer-like ID into the computation graph. The `usize` is not the weight itself. |
| 11–16 | A `Node` stores a computed number (`value`), its eventual gradient, and up to two parent links with local derivatives. |
| 18–20 | `Tape` owns all nodes created during one forward pass. |
| 23–25 | Start with an empty computation graph. |
| 28–30 | `leaf` inserts a number (constant or trainable weight) with no incoming operation. |
| 33–45 | `push` records the result of one operation and the derivatives toward its inputs. |
| 48–54 | Read the forward value and backward gradient of any node. |
| 57–59 | For `a+b`, the derivative toward each input is `1`. |
| 62–64 | For `a-b`, the derivatives are `1` and `-1`. |
| 67–69 | For `a*b`, the derivatives are `b` and `a`. |
| 72–74 | Scaling by a constant multiplies gradients by that constant. |
| 77–84 | Division applies the quotient rule. |
| 87–90 | Exponential's derivative is itself. |
| 93–95 | Natural log's derivative is `1/x`. |
| 98–101 | ReLU's derivative is 1 for positive inputs, 0 otherwise. |
| 104–114 | Set the loss gradient to 1 and traverse the graph backward, adding every gradient contribution to its parents. This is backpropagation. |
| 118–141 | Unit tests for multiplication and a value used twice. |

### Worked example: backpropagation with two numbers

```text
x = 3
w = 4
loss = x * w = 12

Derivative of loss with respect to x = 4
Derivative of loss with respect to w = 3
```

That is the entire idea behind gradient-based learning. A Transformer simply composes many such elementary operations into a larger graph.

## Lesson 5 — `src/model.rs`: loss and weight updates

| Line(s) | Explanation |
|---|---|
| 197–209 | Calculate cross-entropy error for the actual next character. A lower value means the model assigns higher probability to the correct answer. |
| 212–213 | `train_one` receives input character IDs, target IDs, Adam, and learning rate. The two slices must have the same length. |
| 214 | Start a fresh computation graph for this training window. |
| 215 | Put the current weight values into the graph as leaf nodes. |
| 216 | Run all forward computations through the Transformer. |
| 217–221 | Accumulate one prediction error per input position. |
| 222 | Average the loss across the positions in the window. |
| 223 | Save the scalar loss for logging before updating weights. |
| 224 | Run the chain rule backward through the whole graph. |
| 225 | Extract `d(loss)/d(weight)` for every parameter. |
| 226 | Ask Adam to modify all trainable weights. |
| 227 | Return the loss so `main.rs` can print it. |
| 308–312 | Adam stores a moving average of gradients and squared gradients, plus an update counter. |
| 319–337 | Apply gradient clipping, correct Adam's moving-average bias, and subtract a small step from each weight. |

## Lesson 6 — `src/model.rs`: generating text

| Line(s) | Explanation |
|---|---|
| 231–233 | Accept a prompt, requested output length, sampling temperature, and seed. |
| 234–240 | Convert prompt characters into known vocabulary IDs; reject unknown characters. |
| 241–243 | Keep output text and create a seeded sampling generator. |
| 244–245 | For each new character, keep only the most recent 12 IDs. |
| 246–250 | Run the same network without backpropagation, then select logits at the last position. |
| 251 | Divide logits by temperature to control randomness. |
| 252 | Draw a next-character index from the probability distribution. |
| 253–254 | Append the selected character to the context and output string. |
| 256 | Return the completed text. |
| 258–284 | Save/load model weights and the vocabulary in a simple plain-text format. |
| 295–305 | Turn logits into a probability distribution and sample one index. |

## A suggested study session

1. Run `cargo run -- train 1` and set a breakpoint at `main.rs:41`.
2. Step into `Model::train_one` at `model.rs:212`.
3. Inside the forward pass, inspect `tokens`, `x`, `q`, `k`, `v`, and `attention`.
4. When `loss` is computed, enter `Tape::backward` and watch a node's `grad` change.
5. Step through `Adam::update` and compare a single `weights[i]` before and after.
6. Repeat `train 500` and compare its loss behavior against a single update. Learning won't necessarily be monotonic across *different* random windows.

The source is commented at each important operation, so you can trace the math and the corresponding Rust syntax side by side.