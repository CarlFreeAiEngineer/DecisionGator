# Evidence check 0.2 (experimental)

A second model, separate from the 0.4.1 `is_yes`/`choose` model. Give it the text a reply was written from (the evidence) and the reply; it flags, sentence by sentence, clearly new information that the evidence does not hold. Use it to catch a language model inventing things before its reply reaches anyone.

**Better than 0.1, still short of the goal.** On real replies to 36 questions it never trained on (45 clear inventions confirmed by a second reviewer, 1,612 honest messages), with the whole prompt as evidence:

| Flag when p(unsupported) is at least | Clear inventions caught | Honest messages flagged |
| ---: | ---: | ---: |
| 0.3 | 71% | 7.0% |
| 0.5 | 69% | 4.4% |
| 0.7 | 62% | 2.4% |
| 0.9 | 38% | 1.2% |

The goal is 9 in 10 caught at 1 in 20 flagged. The same training gave 0.1's recipe 62% caught at 7.2% flagged on this test. A new detail joined onto an otherwise true sentence ("large muscles and long hair") is caught 98 to 99% of the time on planted test cases. With only 45 real inventions in the test, each one is worth about 2 points.

What changed since 0.1:

- **An alarm now means clearly new information only.** Sentences that tell a known fact wrongly (a changed name, number or order) and borderline ones are not what it looks for; do not rely on it for those.
- It learned by copying a larger model (ModernBERT-large, 76% caught at 4.1% flagged on the same test, about twice as slow), and from about four times as many labeled real replies.

**Speed and evidence length.** Give it the whole text the reply was written from, up to 4,096 tokens. Cutting a 3,500-token prompt to 2,048 tokens removes lines that support honest sentences: caught stays at 67% but honest messages flagged rise to 10.9%. On a 6-core server processor 0.1 (same size and shape) took about 2.5 seconds at 2,000 tokens and about 6 seconds at 3,500; 0.2 was not timed again.

**Very short evidence fails.** It was trained on evidence of about 400 tokens and more. Given under about 100 tokens (a few lines) it flags nearly every sentence, honest or not; from about 120 tokens up it behaves as measured above. 0.1 did not have this fault.

Files:

- `model.onnx`: `MoritzLaurer/ModernBERT-base-zeroshot-v2.0` (Apache-2.0) fine-tuned with `training/evidence/train_message.py --only-new-information --teacher`, float32, 599 MB.
- `tokenizer.json`: its tokenizer.
- `manifest.json`: input format, temperature, and hashes of the model, script and training data.

Trained on the public invented sets in `data/evidence/` plus private labeled replies that are not in this repository, so retraining from the repository alone gives a different, probably weaker, model.

Run it: `uv run examples/evidence_check.py --demo`. It downloads these files from `https://62-84-178-253.sslip.io/DecisionGator/files/evidence-check-0.2/` and checks their hashes. Details: [the report](../../reports/evidence-check-v1.md) and [the design](../../specs/evidence-check.md).
