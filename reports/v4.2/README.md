# Candidate 0.4.2: better routing with short option labels

Version 0.4.1 routed messages poorly when the options were short and generic, such as `["support", "billing", "sales"]`: a bare "support" option absorbed billing problems phrased as requests for help, and receipts or tax questions went to "sales". On a new held-out test ([choices-v2-test](../../data/choices-v2-test.jsonl)) it routed 25 of 42 support cases correctly. [choices-v2](../../data/choices-v2.jsonl) adds 160 training records aimed at this; see [its notes](../../data/choices-v2.md).

Two ways of training were tried on a Colab L4 GPU (an A100 was not available), both with every training file used for 0.4.0 plus `choices-v2.jsonl`, and both compressed like 0.4.1 (`mixed_quant.py ... 6 17 --reduce-range`) and calibrated on the calibration split.

- `retrain/`: the full 0.4.0 recipe from the foundation model (5 passes, learning rate 1e-5). It fixed routing but lost yes/no accuracy, uncompressed as well (90.0% on the 480-case test against 92.3% for 0.4.0), so it was set aside.
- `continued/`: the 0.4.0 checkpoint trained further for 2 passes at learning rate 5e-6; the pass with the lowest validation log loss (pass 1) was kept. This is the candidate.

## Accuracy, compressed models, scored on the Mac

| Test | 0.4.1 | Retrain | Continued |
| --- | ---: | ---: | ---: |
| New test, 480 | 90.6% | 89.6% | 90.8% |
| Fresh test, 80 | 92.5% | 92.5% | 91.2% |
| Original test, 52 | 94.2% | 94.2% | 94.2% |
| Choice test, 20 | 90% | 95% | 100% |
| Choices-v2 test with the choice test, 80 | 76.2% | 93.8% | 93.8% |
| Support routing within it, 42 | 59.5% | 90.5% | 90.5% |
| Spam test, 32 | 93.8% | 81.2% | 96.9% |
| Email test, 40 | 85.0% | 77.5% | 82.5% |

The continued model's fitted temperature is 2.265 (0.4.1: 1.603). Log loss on the 480-case test is 0.210 against 0.213. Uncompressed, the continued model scores 95.2% on support routing (40 of 42).

## Reproduce

```text
uv run decisiongator-train train --output runs/v4.2-ft --start runs/v4-large/best --nli-head --template 2 --learning-rate 5e-6 --epochs 2 --batch-size 16 --seed 42 --device cuda --extra-data data/expansion-v2.jsonl --extra-data data/choices.jsonl --extra-data data/choices-v2.jsonl --extra-data data/v4/appointment_intent.jsonl ... --extra-data data/v4/urgency.jsonl
uv run decisiongator-train export --checkpoint runs/v4.2-ft/best --output models/v4.2-ft-large --model-id decisiongator-0.4.2-large-experimental
uv run --locked python training/quant/mixed_quant.py models/v4.2-ft-large models/v4.2-ft 6 17 --reduce-range
uv run --locked decisiongator-train calibrate --bundle models/v4.2-ft --output reports/v4.2/continued/compressed/calibration.json <same --extra-data list>
```

The `...` stands for every training file in `data/v4/` except `test-*.jsonl`. Held-out tests are scored with `decisiongator-train evaluate --split test` on the same files as [the 0.4.1 report](../accuracy-v4.1.md), plus `data/choices-v2-test.jsonl`. Run configuration and per-pass history are in `continued/` and `retrain/`. No release bundles have been rebuilt from this model yet.
