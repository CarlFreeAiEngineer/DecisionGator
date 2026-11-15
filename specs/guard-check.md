# Guard check: sexual content together with a child

A worked use of DecisionGator as a safety check, built for an adults-only companion chat site (RealAiDating, `../RealAiDating/specs/guard.md`). It replaces a hosted model with two local yes/no calls. Numbers are in [the report](../reports/guard-check.md); the code to copy is [examples/guard_check.py](../examples/guard_check.py).

## The rule

A message is blocked when it has sexual content and also mentions a child or anyone under 18, anywhere in it. Explicit sexual talk between adults passes. Plain talk about children passes. Pet names for an adult ("baby", "good girl", "little man", "daddy") are not a child.

The other half of the site's rule, anything sexual about a person under 18, is the same decision here: such a message has sexual content and mentions a minor, so it is blocked. But no text of that kind was written, generated or stored to train or test this check, so that half is not measured. Keep the writing models' own refusals as the first line there.

## How to ask

Two questions, combined in code. Do not ask one joined question ("a child and also sexual content?"): the model was not trained on it.

1. **Sexual content.** `is_yes_p(piece, "Does this message contain sexual content?")`, on the bare message, no criteria.
2. **A child mentioned.** `is_yes_p(content, "Does the message mention a child or anyone under 18?")`, no criteria. Asked only when the first answer is yes, so most ordinary messages cost one call.

Use exactly these question texts: the model was trained on them. It was also taught the child question with criteria (`CHILD_CRITERIA` in the example), but that form adds about 80 tokens to every call and has not been scored on the test for the trained model; with 0.4.1 the criteria made the answers worse. Leave them out.

**Background for the child question.** "She slept in the next room" breaks the rule only if "she" is a child, so the content for the child question may carry what the application knows, in this exact shape:

```text
Known: Mia is the sender's daughter, aged six.
Before: this is me and my daughter Mia
Message: <the message, or one piece of it>
```

`Known` is one or more short sentences about the children in the people's lives (name, whose child, age). `Before` is the message sent just before, by either side; it is never judged itself. Leave out a line that is empty; with neither line, pass the bare message without the `Message:` label. The model was trained to answer yes when the message refers to a known child by name or by "she", "he" or "they", and to answer no when a child is in the background but the message does not refer to them, so a parent's sexual talk with another adult still passes.

**Thresholds.** Block when sexual is at or above 0.03 and child is at or above 0.20. They are low on purpose: every doubt blocks. The rule set before looking at the test was: on validation messages, half of the lowest answer any must-block message received. That gave 0.03 for sexual (one mildly worded validation message scored 0.065) and 0.49 for child. The child threshold is then set lower still, at 0.20, as the more cautious choice; on the test it costs one more false alarm. A call that fails must block too.

## Long messages

The model reads at most 256 tokens: content, question and criteria together, and it refuses longer input instead of cutting it. So a long message is read in overlapping pieces of whole sentences, about 320 characters each (about 80 tokens), every piece starting with the last sentence of the one before. Each question's answer for the message is the highest answer over its pieces, and the two are then combined, because the rule is about the whole message: sex in one paragraph and a child in another still blocks. Background lines are cut to 160 characters each (the end of `Before`, the start of `Known`), which keeps every call under about 200 tokens. `pieces()` and `check()` in the example do all of this.

Cost: one call per piece for the first question, and one more per piece only when the message is sexual. On an M1 Pro a call takes about 75 to 90 ms: about 0.15 seconds for a short sexual message and about 0.7 seconds for a message of 130 to 220 words.

## What it is not good at

- A child who is referred to without any child word, name or background ("helping Jonah with his long division"). Give it `Known` lines.
- Very mild sexual wording ("take me to bed") scores lower than explicit wording. The low threshold is there for this.
- A writer's own childhood ("when I was eleven") counts as a mention of a person under 18. Alone it passes, since nothing is sexual; in a message that also has sexual content it blocks. That is the cautious reading of the rule, and it will hold some harmless messages.
- Only English, like the rest of the component.

## How it was trained

Version 0.4.3 is the 0.4.0 checkpoint trained further (3 passes at learning rate 5e-6, the second pass kept for its lowest validation loss, the way [candidate 0.4.2](../reports/v4.2/README.md) was made) on every earlier training file plus the guard records, then compressed and calibrated like 0.4.1.

The guard texts are made by `training/guard/make_data.py`: Claude Sonnet wrote the training messages and Claude Opus the test messages (a different writer, different names), through the `claude` command. Claude Haiku then answered both questions for every text itself; a text was kept only when Haiku agreed with what the writer was asked for. Kinds: sexual talk between adults with no child (crude, tender, mild, with pet names and adult ages); plain talk about children, family, school and one's own childhood; sexual talk between adults in which a child is mentioned non-sexually (the class to block); messages that name a person only by first name or pronoun, each with two backgrounds, one making that person a child and one an adult (so the model must read the background); and ordinary talk with neither. `training/guard/build_rows.py` turns them into training records and adds borrowed backgrounds to texts written without one.

The hard limit, stated in every writing prompt and checked by Haiku on every text: nothing sexual about, involving, directed at or witnessed by a person under 18. Texts Haiku flagged were dropped without being saved.

**The texts are not in this repository.** They are explicit, so they live in `data/local/guard/`, which Git ignores. This model therefore cannot be rebuilt from the public data alone; the scripts rebuild a similar set. Whether to publish the texts is the owner's decision.

```text
uv run training/guard/make_data.py train --model sonnet      # writes data/local/guard/items-train.jsonl
uv run training/guard/make_data.py test --model opus         # writes data/local/guard/items-test.jsonl
uv run training/guard/build_rows.py                          # train.jsonl (records), validation.jsonl, test.jsonl
uv run decisiongator-train train --output runs/NAME --start runs/v4-large/best --nli-head --template 2 --learning-rate 5e-6 --epochs 3 --batch-size 16 --seed 42 --device cuda <the --extra-data list of reports/v4.2> --extra-data data/local/guard/train.jsonl
uv run decisiongator-train export --checkpoint runs/NAME/best --output models/NAME-large --model-id decisiongator-0.4.3-large
uv run --locked python training/quant/mixed_quant.py models/NAME-large models/NAME 6 17 --reduce-range
uv run decisiongator-train calibrate --bundle models/NAME --output runs/NAME/calibration.json <the same --extra-data list>
uv run training/guard/evaluate.py score --bundle models/NAME --name NAME
uv run training/guard/evaluate.py report 0.4.1 NAME --table
uv run code/rebundle.py --target macos-arm64 --model models/NAME --output models/0.4.3-macos-arm64 --model-id decisiongator-0.4.3
uv run code/build_python.py --target macos-arm64 --bundle models/0.4.3-macos-arm64 --version 0.4.3
```

Training ran on a Colab A100, about 8 minutes a pass. Nothing was trained on the Mac.
