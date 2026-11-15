# Evidence check, second round: new information only (2026-11-15)

Follows [the first report](evidence-check-v0.md). Not released yet. Scores on private real replies are numbers only.

## What changed

- **A narrower target.** An alarm should mean a reply states clearly new information that its evidence does not hold (the evidence says "Arnold has big muscles", the reply says "large muscles and long hair"). Retelling a real event with a wrong day, order, number or person ("told wrong"), and borderline rewording or elaboration, are out of scope: such sentences carry no label in training or scoring (`train_message.py --only-new-information`).
- **Much more real material.** 60 more batches of replies from Hy3 and DeepSeek to the same 106 prompts (OpenRouter, $2.74 in all). 28 batches were labeled sentence by sentence with Sonnet, which also says whether an unsupported sentence is new information or told wrong: 13,943 labeled real sentences from 4,848 replies. 32 batches are unlabeled, for the small model to learn from the large one.
- **A stricter test.** 36 of the 106 questions are test only: no reply to them from any model is trained on. Validation and calibration use 10 questions each.
- **A second review.** Opus re-judged, without seeing the first label or the model's score, every real sentence Sonnet called new information (train and test, clear and borderline) and every test sentence where the first model and the label disagreed. It confirmed about half of Sonnet's "clear" inventions and almost none of its false alarms. Clear inventions are rare: 45 in 1,657 test replies (about 1 in 37).
- **Planted additions.** `add_real.py` joins one clearly new detail onto a real supported sentence (2,523 kept after a Haiku check, private), and `make_additions.py` does the same on invented material (public, `data/evidence/additions.jsonl.gz`, 700 items, 1,975 additions and 654 rewordings that add nothing).
- **A small model that copies a large one.** `train_message.py --teacher` adds a term that matches the large model's probabilities at every sentence, labeled or not.

## Results

Real replies to the 36 held-out questions, clear new-information inventions confirmed by the second review (45 sentences; one sentence is 2 points) and messages without one (1,612). Alarm when p(unsupported) is at least the threshold. Whole prompt as evidence (4,096-token limit).

| Model | Threshold | Inventions caught | Clean messages flagged |
| --- | ---: | ---: | ---: |
| Base | 0.5 | 62% | 7.2% |
| Base, copying large | 0.5 | 69% | 4.5% |
| Base, copying large, plus unlabeled replies | 0.5 | 69% | 4.4% |
| Base, copying large, plus unlabeled replies | 0.9 | 38% | 1.2% |
| Large | 0.3 | 80% | 6.1% |
| Large | 0.5 | 76% | 4.1% |
| Large | 0.9 | 64% | 1.4% |

By writer, large model at 0.5: Hy3 65% caught (17 inventions) with 2.3% flagged; DeepSeek 79% (19) with 1.4%; Claude 83% (6) with 8.9%.

Planted additions in real test replies (521): the large model catches 99% at 0.5, base 98%. Invented test sets stay at 97 to 100% for added details and never-told facts.

The goal in REDESIGN.md was 90% caught with at most 5% flagged. The large model reaches about 78% at 5%; the small one about 70%.

## Trimming the evidence costs accuracy

The prompts here run to about 3,500 tokens. Cutting the evidence from the front to fit a smaller limit removes lines that support honest sentences, so they raise alarms:

| Model at 0.5 | Limit | Inventions caught | Clean messages flagged |
| --- | ---: | ---: | ---: |
| Base, copying large | 4,096 | 69% | 4.4% |
| Base, copying large | 2,048 | 67% | 10.9% |
| Large | 4,096 | 76% | 4.1% |
| Large | 2,048 | 80% | 13.8% |

So the 2.5 seconds measured for the base model at 2,000 tokens is not the cost for this application. With the whole prompt it is about 6 seconds for base and 13.5 for large on a 6-core Xeon ([first report](evidence-check-v0.md)), against a budget of 1 to 3.

## What did not help

- Cleaner training labels alone (base 62% at 7.2% after, 65% at 11% before, on slightly different test labels).
- 4,141 unlabeled real sentences for the small model to learn from, beyond copying the large model on the labeled data.

## Limits

- 45 confirmed inventions is a small test. The second review covered every sentence labeled unsupported and the first base model's false alarms, not every false alarm of the later models, so their flagged rates may include a few real inventions the first label missed.
- The test questions are new to the model, but the chat history behind them is the same one the training questions use.
- Labels come from language models (Sonnet, then Opus), not from people.
