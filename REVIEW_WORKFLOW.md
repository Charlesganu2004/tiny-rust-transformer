# How to study and review each pull request

This project is a guided code-review course. None of the eight pull requests needs to be merged before you study it.

## Read and annotate
1. Open the PR links in the README, starting from Lesson 1.
2. Open **Files changed** and click the blue **+** beside a line.
3. Write what the line does **in your own words**, the Rust syntax you recognize, the math behind it, or an uncertain point.
4. Click **Start a review** to collect comments, then **Finish your review** and **Submit review**. Keep question threads open until you have checked the explanations.
5. To experiment, checkout the PR's head branch, edit the source, run `cargo test`, and push your commits to that branch.

## How to ask for feedback
Message ChatGPT: **Review PR #3 in Charlesganu2004/tiny-rust-transformer, read my inline comments and new commits, correct mistakes, and explain every correction.**

A response should:
- Distinguish **correct / partially correct / incorrect** explanations.
- Show the actual Rust behavior with a small worked example.
- Reply within the relevant GitHub review thread when the connected account is allowed to.
- Propose or commit fixes to the **same learning branch** if you ask for corrections.
- Never auto-merge, silently rewrite your explanation, or delete your comments.

## Stacked PR bases
These are *stacked*: PR #2 targets PR #1's branch; PR #3 targets PR #2's branch; and so on. This keeps each review small. While unmerged, the complete implementation is visible on `learn/08-tests-and-study-guide`. If a lower branch changes after a review, upper branches may need rebasing or updating. Ask me to keep the stack synchronized if necessary.

## Review checklist
- Can I state the inputs and outputs of this function?
- Can I explain why each borrow is `&` or `&mut`?
- What is the data type of each intermediate variable?
- What shape is each vector/matrix (and why)?
- What mathematical derivative or probability operation does it represent?
- What test would fail if I broke this behavior?

The implementation is a miniature educational model, not a production chatbot or a foundation LLM.
