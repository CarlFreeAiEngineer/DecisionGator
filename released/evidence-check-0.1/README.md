# Evidence check 0.1 (experimental)

A second model, separate from the 0.4.1 `is_yes`/`choose` model. Give it the text a reply was written from (the evidence) and the reply; it says, sentence by sentence, whether the evidence supports it. Use it to catch a language model inventing things before its reply reaches anyone.

**Barely useful yet.** On real replies from the model that invents most (Hy3) it catches about half of the clear inventions while flagging about 1 in 25 honest messages. The goal is 9 in 10 caught at 1 in 20 flagged. On invented test material it scores 96 to 98%, so it has learned the task; it needs much more real training data. Details: [the report](../../reports/evidence-check-v0.md) and [the design](../../specs/evidence-check.md).

- `model.onnx`: `MoritzLaurer/ModernBERT-base-zeroshot-v2.0` (Apache-2.0) fine-tuned with `training/evidence/train_message.py`, float32, 599 MB.
- `tokenizer.json`: its tokenizer.
- `manifest.json`: input format, temperature, and hashes of the model, script and training data.

Trained on the public invented set in `data/evidence/` plus private labeled replies that are not in this repository, so retraining from the repository alone gives a different, probably weaker, model.

Run it: `uv run examples/evidence_check.py --demo`. It downloads these files from `https://62-84-178-253.sslip.io/DecisionGator/files/evidence-check-0.1/` and checks their hashes. On a 6-core server processor one call over 2,000 tokens takes about 2.5 seconds.
