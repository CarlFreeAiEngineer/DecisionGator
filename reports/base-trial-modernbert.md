# ModernBERT trial: not adopted for the short questions

On 2026-11-14 `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` (revision `a51e07b5`, Apache-2.0, reads up to 8,192 tokens) was trained with the unchanged 0.4 recipe in place of DeBERTa-v3-large, to see whether one long-input model could serve both today's questions and the evidence check in [REDESIGN.md](../REDESIGN.md). It could not match DeBERTa on the short tests, so `is_yes` and `choose` stay on DeBERTa and ModernBERT is used only for the evidence check ([specs/evidence-check.md](../specs/evidence-check.md)).

## Accuracy

Uncompressed models, temperature fitted on the calibration split, same held-out tests as [the 0.4.1 report](accuracy-v4.1.md). Trained on a Colab L4 with `training/base_trial.py` (learning rate 1e-5, batch 16, seed 42).

| Test | 0.4.0 DeBERTa | ModernBERT, pass 1 of 5 | ModernBERT, pass 3 of 3 |
| --- | ---: | ---: | ---: |
| New test, 480 | 92.3% | 86.2% | 86.7% |
| Fresh test, 80 | 92.5% | 92.5% | 91.2% |
| Original test, 52 | 94.2% | 88.5% | 92.3% |
| Choice test, 20 | 95% | 90.0% | 90.0% |
| Spam test, 32 | 90.6% | 75.0% | 71.9% |
| Email test, 40 | 82.5% | 82.5% | 85.0% |

The recipe keeps the pass with the lowest validation log loss, which for ModernBERT was always pass 1, the least accurate one, so a second run stopped after 3 passes and scored the last pass. It made little difference. Validation accuracy peaked at 89 to 91% in both runs against 93 to 95% for DeBERTa; by pass 3 the model was right on 98.8% of its training data while its validation log loss kept rising, so it memorizes these 3,674 examples rather than generalizing from them.

This is the second ModernBERT-family result below DeBERTa; [the Laya trial](laya/README.md) reached 80% on the new test.

## What might close the gap, not tried

- Training ModernBERT to copy DeBERTa 0.4.0's probabilities on many unlabeled questions.
- Weight decay, a lower learning rate, and choosing the pass by accuracy.
- More training data of the same kind, such as the evidence-check records.
