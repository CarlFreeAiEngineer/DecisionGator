# Evidence check, first model (2026-11-15)

First three-way model for the evidence check described in [specs/evidence-check.md](../specs/evidence-check.md). Not released. Scores on the private real-reply records are given as numbers only.

## Training

`MoritzLaurer/ModernBERT-large-zeroshot-v2.0` with a three-way head, `training/evidence/train.py`, 3 passes at learning rate 1e-5, unsupported weighted 2x, 4,096 tokens, on a Colab A100 (about 55 minutes with flash attention). Data: the public invented set (`data/evidence/invented.jsonl.gz`, 17,401 sentences), plus 4,086 sentences labeled from RealAiDating's real replies and 846 of them with their support removed (both private, not in this repository). The 277 test sentences (all from Hy3) were relabeled by a careful second review; 25 labels changed.

## Accuracy

| Test | Sentences | Accuracy | Unsupported caught |
| --- | ---: | ---: | ---: |
| Invented, held out | 921 | 98.2% | 97.6% |
| Invented, a subject never trained on | 1,237 | 97.7% | 96.6% |
| Real, support removed (validation) | 71 | 100% | 100% |
| Real Hy3 replies (test) | 277 | 85.2% | 55% |

On the real Hy3 test, flagging a sentence when p(unsupported) is at least the threshold:

| Threshold | Invented sentences caught (of 20) | Messages with no invention flagged (of 88) |
| ---: | ---: | ---: |
| 0.3 | 60% | 10% |
| 0.5 | 50% | 6% |
| 0.7 | 40% | 3% |

Counting only the 13 inventions the reviewer marked clear: 62% caught at 0.3 with 4% of clean messages flagged. The goal in REDESIGN.md is 90% caught with at most 5% flagged, so this model is not yet good enough on real replies, even though it is near perfect on invented ones.

## Speed

ONNX Runtime 1.22.1 on the M1 Pro, 8 threads, one sentence per call:

| Tokens | Float32 | 8-bit weights |
| ---: | ---: | ---: |
| 512 | 0.67 s | 0.52 s |
| 1,000 | 1.5 s | 1.2 s |
| 2,000 | 4.3 s | 3.4 s |
| 3,500 | 8.9 s | 7.5 s |

Compressing the weights barely helps, because at these lengths the attention dominates: the exported model computes full attention in every layer, although ModernBERT's design limits two layers in three to a window of 128 tokens. A message of four sentences against a 3,500-token prompt takes about 35 seconds, against a budget of 1 to 3.
