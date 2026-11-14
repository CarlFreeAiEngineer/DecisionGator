# Redesign note: checking a sentence against a long text

Written 2026-11-14 from work on RealAiDating (`../RealAiDating`, [specs/memory.md](../RealAiDating/specs/memory.md)). This is a request and the evidence behind it, not a finished design. DecisionGator should support this in addition to what it does today (`is_yes`, `choose`), which stay as they are.

## The need

A language model writes a message for a character who has a long history with a person. The prompt holds what the character has in mind: recalled memories, the recent conversation, a short list of what weighs on them, a description of who they are. The model sometimes states something about the past that the prompt does not hold: it named a father "Elias" when no name was ever given, said an event happened "the day after" when it was the same evening, added "cursing the drill" to a true memory of a dentist visit. People rarely invent in detail like this, so it is the main thing that gives a model away over a long relationship.

On RealAiDating's memory test (106 questions after an 8-week chat), the live model (Hy3) made 12 false or invented replies of 106, GPT-6 Luna 8, DeepSeek 15, Claude Sonnet 8. Changing the model does not fix it. A clearer prompt took Claude Haiku from 12 to 1 but did not fix Hy3.

The fix we want: before a message is sent, check each sentence against what the character had in mind. A flagged message is written once more with the unsupported sentence named.

## What is asked of DecisionGator

One new capability: given a short text (one sentence, or a few) and a long text (the evidence), say whether the short text states anything about the past that the long text does not support.

- **Input:** the sentence, and evidence of up to about 2,000 tokens. Today's limit is 256 tokens in all (`max_tokens` in the bundle manifest), which is the main obstacle.
- **Output:** probabilities for three answers: supported; not supported; not a claim about the past (a question, a feeling, something happening now, a plan).
- **Where it runs:** locally on a CPU, like everything else here. RealAiDating's server has 8 AMD EPYC cores and no GPU. One check per message of 1 to 3 seconds is affordable, since the reply it checks takes 10 to 40 seconds to write.
- **Shape of the call:** open. It might be `is_yes(evidence, question)` with a long-input model and a fixed question, a new function such as `supported_p(evidence, statement)`, or a three-way `choose`. Whatever fits the component best. The caller sets the threshold, as now.

What it does not need to do: judge truth. It checks whether the evidence supports the sentence, nothing more. A true fact the evidence does not hold should be flagged (the character should then say "I think you told me, remind me?").

This is useful beyond RealAiDating: checking any generated summary or answer against its source is the same job.

## What we measured

**DecisionGator 0.4.1 as it is, untrained for this.** On the 106 Haiku replies of the run with 12 invented, each sentence was first asked "does this sentence state a fact about something that happened or was said before now?", then checked with "is what the sentence says about the past supported by the evidence?" against the few memories nearest the sentence that fit in 256 tokens. About 0.1 seconds per call.

| Threshold on p(supported) | Replies flagged, of 106 | Invented replies caught, of 12 |
|---|---|---|
| below 0.2 | 16 | 4 |
| below 0.3 | 35 | 6 |
| below 0.5 | 64 | 9 |

It pointed at the right sentence in several cases ("I remember you biting your cheek and cursing the drill" 0.15; "You carried a whole house on them this weekend" 0.14), so the shape works. The false alarms had two causes: evidence that had to be left out to fit 256 tokens ("Seattle, you told me" was true), and questions or present-tense sentences getting past the first question ("How did the interview go?").

In six hand-made cases the positive question worked far better than the negative one ("does the reply state anything the evidence does not say?" was near chance).

**Jev, for comparison.** One call for the whole reply against the whole prompt (about 4,000 tokens), 0.24 seconds: caught 8 of 12 but flagged 48 of 106. Too blunt at whole-reply level; not tried sentence by sentence.

**Speed of a long-input foundation.** ModernBERT reads up to 8,192 tokens. Untrained, PyTorch, 8 threads on an M1 Pro, one forward pass:

| Model | Parameters | 256 tokens | 1,000 tokens | 2,000 tokens |
|---|---|---|---|---|
| ModernBERT-base | about 150M | 0.12 s | 0.38 s | 1.0 s |
| ModernBERT-large | about 400M | 0.25 to 0.45 s | 0.7 to 0.8 s | 1.8 to 2.2 s |

Not measured: the ONNX runtime, quantized weights, or the server's CPU. The current foundation (DeBERTa-v3-large) was pretrained on 512 tokens, so stretching it to 2,000 is likely to cost accuracy; that is a guess, not a measurement.

## Training data to start from

All in `../RealAiDating/evals/memory/` (the run folders are not in git; they are on the Mac that made them, and `app.py` makes more):

- `runs/app-*/prompts/P*.md`: the evidence, exactly what the character had in mind for each reply (everything above "# What to write").
- `runs/app-*/replies.json`: the reply to each of 106 questions, from Haiku (5 runs), Sonnet, Hy3, Luna and DeepSeek: about 950 replies, several sentences each.
- `runs/app-*/verdicts.json`: a judge's verdict per reply (yes, partly, no, wrong) with a short reason that often names the invented part. Verdicts are per reply, not per sentence, and "wrong" was judged against the whole chat, not against the prompt, so sentence labels have to be made from them (a language model can do that, with a person checking a sample).
- `history/week-*.jsonl` and `plan.yaml`: the 8-week chat, 1,279 messages, with about 60 planted facts. The bulk of the training set can be generated from it: take a true statement about the chat and alter a name, a number, a date, the order of two events, who said it, or add a plausible detail. These are the kinds of error the models actually make.

Wanted in the set, so the model does not learn shortcuts: supported sentences that paraphrase the evidence in quite different words; unsupported ones that reuse the evidence's words; honest forgetting ("did you tell me? I have forgotten"), which is supported; evidence where the supporting line is anywhere in the 2,000 tokens; and other subjects than this one chat, so it works for other characters and other uses.

## How we will know it works

On Hy3 replies kept out of training, judged sentence by sentence: catch at least 9 in 10 invented sentences while flagging no more than 1 in 20 messages that invent nothing. Then, wired into RealAiDating, the memory test on Hy3 should go from 12 false or invented replies of 106 to 2 or fewer without the score (80.5) falling.

## Open questions

- One model for everything, or a second model beside the current one? A long-input model that also answers today's short questions as well as 0.4.1 would be simplest for callers; two models double the bundle size.
- ModernBERT base or large? Base is twice as fast; accuracy is unknown until trained.
- Does the "not a claim about the past" answer belong in the model, or in a separate cheap first call as in the trial above?
- Sentence by sentence (several calls per message) or the whole message at once with the model pointing at the worst sentence?
- How RealAiDating should take it: the Python wheel today is for macOS arm64, Linux x64 and Windows x64; the server is Linux x64.
