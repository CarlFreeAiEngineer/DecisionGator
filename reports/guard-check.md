# Guard check: version 0.4.3 against 0.4.1

Measured 2026-11-15. What the check is and how to call it: [specs/guard-check.md](../specs/guard-check.md). A message must be blocked when it has sexual content and also mentions a child or anyone under 18.

## Result

Held-out test: 1,098 messages written by a different model than the training messages (952 different texts; some appear twice with different background). 340 must be blocked, 758 must pass. Each message is judged whole, as `examples/guard_check.py` does it. A miss is a message that should be blocked and passed; a false alarm is one that should pass and was blocked.

| Model | Thresholds (sexual, child) | Misses, of 340 | False alarms, of 758 |
| --- | --- | ---: | ---: |
| 0.4.1 | 0.02, 0.11 (its own, from validation) | 2 | 176 |
| 0.4.1 | 0.5, 0.5 | 27 | 85 |
| 0.4.3 | 0.03, 0.49 (from validation) | 0 | 7 |
| 0.4.3 | 0.03, 0.20 (recommended) | 0 | 8 |
| 0.4.3 | 0.5, 0.5 | 0 | 2 |

Each question alone on the test, 0.4.3 at 0.03 and 0.49: the sexual question is wrong on 17 of 1,098 messages, the child question on 3. For 0.4.1 at its own thresholds: 36 and 237.

The 12 messages of RealAiDating's first trial (the regression list, 4 to block and 8 to pass, the first one with its background lines): 0.4.3 gets 12 of 12, through the Mac wheel and through the Linux wheel. 0.4.1 at its own thresholds blocks all 4 and wrongly blocks 3 of the 8.

By kind, 0.4.3 at 0.03 and 0.49, wrong of total: sexual talk between adults with no child 1 of 210; sexual talk with a child mentioned 0 of 226; plain talk about children 3 of 165; a person known as a child only from the background, sexual message 0 of 83, plain message 2 of 63; the same person known as an adult, sexual 0 of 88, plain 0 of 64; ordinary talk 0 of 80; long messages (130 to 220 words): to block 0 of 31, sexual with no child 0 of 40, plain with a child 1 of 16, plain 0 of 32.

The false alarms are non-sexual messages about a child that the sexual question scored a little above 0.03 ("before I settle in with you"), and one sexual message between adults with "lad" and "my husband" in it.

## How far to trust this

- Zero misses in 340 does not mean a zero miss rate. With no miss seen in 340, the true rate is below about 1 in 100 with 95% confidence, on messages like these.
- All texts were written by Claude models. Real messages from the site's own writing model and its users may differ. The check should run beside the hosted check for a while, and every disagreement should be looked at, before it replaces it.
- The labels are what the writer was asked for and a second model (Haiku) agreed with. No person has reviewed them.
- Nothing sexual about a minor was written or stored, so that half of the site's rule is not measured here at all.
- The test was scored twice. After the first training round it showed that the sexual question had started to say yes to warm, non-sexual messages with narration and pet names (20 false alarms at the same thresholds, no miss). More training texts of that kind, mildly worded sexual texts and long messages were then written (never test texts), and the model was trained again from the same start. The test set did not change, but the second round was shaped by what the first one showed, so its false-alarm count is somewhat optimistic. The validation messages showed the same weakness.
- A child referred to with no child word, no age and no background is not seen: "helping Jonah with his long division" scored 0.01 on the child question. That message was not sexual, so it was not a miss, but a sexual one like it would be.

## Speed

Through the Python wheel on an M1 Pro (10 cores): 75 ms per call for a short message (95th percentile 107 ms), 87 ms for a piece of a long one. A short message costs one call when it is not sexual and two when it is: median 0.14 seconds. A message of 130 to 220 words is read in 3 to 6 pieces: median 0.70 seconds, slowest 0.89. On a Colab Linux machine (Xeon 2.2 GHz, 12 threads) the Linux wheel took 82 ms per call. RealAiDating's own server has not been timed.

## The general tests

The same held-out tests as [the 0.4.2 candidate report](v4.2/README.md), compressed model, scored on the Mac.

| Test | 0.4.1 | 0.4.2 candidate | 0.4.3 |
| --- | ---: | ---: | ---: |
| New test, 480 | 90.6% | 90.8% | 90.4% |
| Fresh test, 80 | 92.5% | 91.2% | 96.2% |
| Original test, 52 | 94.2% | 94.2% | 94.2% |
| Choice test, 20 | 90% | 100% | 95% |
| Choices-v2 test with the choice test, 80 | 76.2% | 93.8% | 96.2% |
| Spam test, 32 | 93.8% | 96.9% | 90.6% |
| Email test, 40 | 85.0% | 82.5% | 90.0% |

One case is 3 points on the spam test. Nothing here suggests the guard training harmed the general model.

## Data

Training side: 4,881 checked items (4,439 different texts) written by Claude Sonnet, giving 8,891 records: 7,117 for training, 1,059 for validation, 715 for calibration, about half for each question. Test side: 1,098 items written by Claude Opus with different names. Claude Haiku answered both questions for every text; texts where it disagreed with the writer's brief were dropped (a few in every hundred). Haiku also flagged 3 texts as possibly sexual about a minor in the second round (the count for the first round was lost when the run was stopped); those were dropped without being saved.

The texts are in `data/local/guard/`, which Git ignores. They exist only on the Mac where they were made.

## Files

- Wheels: `released/python/decisiongator-0.4.3-py3-none-macosx_14_0_arm64.whl` and `decisiongator-0.4.3-py3-none-linux_x86_64.whl` (not in Git, not yet published to the download server). They hold the 0.4.1 native library and runtime unchanged, with the new model (`code/rebundle.py`).
- Model: `models/guard-b/` (not in Git). The uncompressed checkpoint is only on the Colab machine that trained it (`runs/guard-b/best`, 1.7 GB) and is lost when that machine is released.
- [guard-check/](guard-check/): training configuration and history, every probability behind the tables above (`scores-*.json`, no message text), and the general test results. [python-wheel-0.4.3-macos-arm64.json](python-wheel-0.4.3-macos-arm64.json) is the wheel check on the Mac.
