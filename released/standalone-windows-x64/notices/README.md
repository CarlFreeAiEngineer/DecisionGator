# Third-party materials

Project source is Apache-2.0 (see the top-level `LICENSE`). Original example data is offered under the CC0 terms described in `data/README.md`. Fine-tuned weights retain their base model's terms; the training recipe and exact source checkpoint are recorded in each bundle's `manifest.json` and in the reports.

## Version 0.4.0 and 0.4.1 foundation (current)

Version 0.4.1 uses the 0.4.0 weights stored as 8-bit and 4-bit integers; no weights were retrained. The 0.4.0 weights are derived from `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` at revision `cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2`, whose model card declares MIT. That checkpoint is itself a fine-tune of `microsoft/deberta-v3-large` (revision `64a8c8eab3e352a784c658aef62be1662607476f`), also MIT. DecisionGator keeps its entailment head and fine-tunes the weights on the published project data. Included here:

- `deberta-v3-large-zeroshot-v2.0-model-card.md`: the zero-shot checkpoint's model card at the revision used.
- `DeBERTa-v3-large-model-card.md`: the Microsoft DeBERTa-v3-large model card.
- `DeBERTa-MIT.txt`: the MIT license from Microsoft's DeBERTa repository. Neither Hugging Face repository ships a separate license file; both declare `license: mit` in their model cards.

Training-data caveat, quoted from the zero-shot model card: models without `-c` in the name, which includes this one, "were trained on more data and perform better, but include data with non-commercial licenses. Legal opinions diverge if this training data affects the license of the trained model." The card recommends the `-c` variants (`deberta-v3-large-zeroshot-v2.0-c`, trained only on commercially-friendly data) for users with strict legal requirements. DecisionGator's own added training data is synthetic and CC0. Users with strict requirements should evaluate this caveat themselves or ask for a retrain on the `-c` checkpoint.

## Earlier foundations

Versions 0.2.0 and 0.3.0 were derived from `cross-encoder/nli-MiniLM2-L6-H768` at revision `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d`, whose model card (`NLI-MiniLM2-model-card.md`) declares Apache-2.0 (`Apache-2.0.txt`). Version 0.1 used `microsoft/MiniLM-L12-H384-uncased` at revision `44acabbec0ef496f6dbc93adadea57f376b7c0ec`, MIT (`MiniLM-MIT.txt`, `MiniLM-model-card.md`). These files are kept for the preserved older bundles under `models/`. No claim is made that this repository contains the upstream pretraining or NLI training corpora.

## Runtime and build dependencies

Each native bundle includes ONNX Runtime 1.22.1's MIT license and third-party notices, and `code/build.py` collects license and notice files for the locked Rust dependencies into `notices/rust-crates/`. The browser bundle carries ONNX Runtime Web's notices. Review that inventory before a public distribution; this is not a completed release audit.
