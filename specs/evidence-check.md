# Evidence check: does a long text support a sentence?

Status: being built (started 2026-11-14). Results so far: [reports/evidence-check-v0.md](../reports/evidence-check-v0.md). The request and its evidence are in [REDESIGN.md](../REDESIGN.md). This page holds the decisions made so far; change it as the work teaches better.

## Decisions

- **A second model, beside the current one.** `is_yes` and `choose` stay on the DeBERTa model (0.4.1). The evidence check gets its own long-input model. Reason: DeBERTa takes 9.3 seconds at 2,000 tokens on the M1 Pro (0.4.1, ONNX Runtime, 8 threads, measured 2026-11-14), and ModernBERT with the 0.4 recipe scored below DeBERTa on the short tests ([report](../reports/base-trial-modernbert.md)). The two can be merged later if a long-input model matches 0.4.1 on the short tests.
- **Foundation:** `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` (Apache-2.0, reads up to 8,192 tokens, already trained on "does this text support this statement" data). ModernBERT-base is the fallback if large is too slow on the server.
- **One call, three answers.** The model has a three-way head: `supported`, `unsupported`, `not_claim` (a question, a feeling, something happening now, a plan). One pass over the evidence gives all three probabilities, so there is no separate first call. The caller sets the threshold.
- **It checks support, not truth.** A true fact the evidence does not hold is `unsupported`. Honest forgetting ("did you tell me? I have forgotten") is `supported`.
- **Input:** the evidence (up to about 2,000 tokens, trimmed by the caller), the whole short text for context, and the one sentence to judge. A sentence alone is often unreadable ("You buried his collar by the fence"), so the model sees the reply it came from. Token limit 2,560 in all.
- **Sentence by sentence.** The caller splits the message and makes one call per sentence. This points at the sentence to rewrite, and it is what worked in the trial in REDESIGN.md.

## Training records

One JSON object per line in `data/evidence/*.jsonl`:

```json
{"id": "...", "group_id": "...", "evidence": "...", "context": "...", "statement": "...", "label": "supported", "kind": "paraphrase", "source": "...", "split": "train", "license": "CC0-1.0", "provenance": {...}}
```

- `label`: `supported`, `unsupported` or `not_claim`.
- `kind`: how the record was made, for scoring by kind. For unsupported records: `name`, `number`, `date`, `order`, `speaker`, `added_detail`, `never_told`, and for flipped evidence also `removed` (the supporting line deleted). For supported ones: `verbatim`, `paraphrase`, `forgetting`. Records from real model replies use `real`.
- `group_id`: records that share evidence or a statement share a group, and a group never crosses splits.
- `split`: `train`, `validation`, `calibration` or `test`.

## Private data

The released model is trained on more data than this repository holds. The sentences labeled from RealAiDating's replies contain that project's private chat, so they live in its private repository (`../RealAiDating/evals/memory/evidence-labels/real-replies.jsonl.gz`, written by `training/evidence/label_replies.py`) and never here. Everything else (scripts, invented data, results as numbers) is public. Anyone retraining the evidence check from this repository alone will get a different, probably weaker, model: add a lot of your own labeled examples before retraining it. Never quote a private sentence in a public report; give scores only.

## Where the records come from

1. **Real replies (private)** from `../RealAiDating/evals/memory/runs/app-*`: about 950 replies to 106 questions by five models. A language model splits each reply into sentences and labels each against the prompt the reply was written from (everything above "# What to write"), using the judge's verdict as a hint. A person checks a sample. All Hy3 replies are the held-out test and are never trained on.
2. **Made from the 8-week chat** (`history/week-*.jsonl`, `plan.yaml`): take a true statement and alter a name, a number, a date, the order of two events, who said it, or add a plausible detail.
3. **Other subjects**, so the model reads the evidence instead of memorizing one chat: other invented people and histories, and source-and-summary pairs.

Guards against shortcuts:

- **Flip the evidence, keep the sentence.** For many records the same sentence appears twice, once with evidence that supports it and once with the supporting line altered or removed. Only reading the evidence can tell them apart. This matters most because the training replies and the test replies come from the same chat.
- Supported sentences that paraphrase in quite different words; unsupported ones that reuse the evidence's words.
- The supporting line placed anywhere in the evidence, start to end.
- A second held-out test on a subject that appears nowhere in training.

## How we will know it works

From REDESIGN.md: on the held-out Hy3 replies, sentence by sentence, catch at least 9 in 10 invented sentences while flagging no more than 1 in 20 messages that invent nothing. Then in RealAiDating the memory test on Hy3 should go from 12 false or invented replies of 106 to 2 or fewer, with the score (80.5) not falling.

Speed: one check of 2,000 tokens in 3 seconds or less on the 8-core server, measured with the ONNX model that ships.

## Still open

- The name and shape of the call in the component (`supported_p(evidence, statement)` or similar) and how the second model is packaged.
- Whether compression like 0.4.1's works for ModernBERT; the 0.4.1 recipe was tuned by hand for DeBERTa.
- ModernBERT base or large, once both are trained and timed.
