# Open seed data

**Found a wrong answer?** Add it to [corrections.jsonl](corrections.jsonl), the one file meant for hand-written fixes; [WRONG-ANSWER.md](../WRONG-ANSWER.md) explains the format and how to retrain. `training/retrain.py` always trains on it. The rest of this page describes the project's own training data.

Version 0.4.0 trains on everything below plus [data/v4](v4/README.md): 3,526 newly authored records in sixteen families and a separate 480-case held-out test. See [the v0.4 report](../reports/accuracy-v4.md).

The current v0.2 recipe also explicitly loads [expansion-v2.jsonl](expansion-v2.jsonl), bringing training/validation/calibration counts to 404/96/52. [evaluation-v2.jsonl](evaluation-v2.jsonl) is a separate frozen 80-example test, never training input. See [the current recipe](../specs/accuracy-v2.md). All these records remain synthetic and unreviewed.

[choices.jsonl](choices.jsonl) adds 120 multiple-choice records (`label_type: choice`, an `options` list, and an index `label`) for the `choose` call; see [its note](choices.md). The pipeline expands each into one yes/no row per option using the runtime's choice template. Loaded with `--extra-data data/choices.jsonl`; not part of the v0.2 release's training.

Additional contributions are opt-in: [email-address corrections](contributions/email-address-correction.md) and [spam-like marketing comments](contributions/spam-comments.md). The spam material includes explicit-policy pairs for borderline invitations and a separate held-out test. Adding files does not automatically load them into training or change the released component; use the documented `--extra-data` arguments.

The input field is now named `content` (previously `state`). This was a field-name-only migration: evidence, questions, criteria, labels, and all other record values are unchanged. Historical training reports retain the original file hashes; [the migration record](../reports/content-field-rename.json) maps those to the renamed files. No retraining was needed. Some original criteria/provenance text still uses the ordinary word “state”; that text is preserved to keep model inputs identical.

The default training pipeline reads both `seed.jsonl` and `plain-questions.jsonl`: 364 records in total (224 train, 56 validation, 32 calibration, 52 test). `contributions/example-correction.jsonl` is a separate worked example loaded only when explicitly passed with `--extra-data`; it is not part of the shipped pilot's training data.

`seed.jsonl` contains 264 original synthetic English examples authored by an AI assistant on 2026-09-17. They are a small development seed for proving the training and native-library workflow, not a reviewed benchmark or evidence of useful general accuracy. No customer records or scraped examples were used. Every record is marked `synthetic_unreviewed`; labels and rationales need independent human review.

On 2026-09-25 the `provenance.generator` field in every data file was changed to a neutral value ("AI assistant" or "AI sub-agent"). Nothing else changed, but the data hashes recorded in `released/*/manifest.json` for earlier releases refer to the files before that edit.

The project intends these original examples and their authored source in `build_seed.py` to be available under CC0-1.0, to the extent rights in the generated material can be dedicated. The CC0 terms are at <https://creativecommons.org/publicdomain/zero/1.0/legalcode>. This notice does not cover upstream model training data or third-party datasets.

| Split | Examples | Yes | No | Purpose |
| --- | ---: | ---: | ---: | --- |
| train | 160 | 80 | 80 | Fit model weights |
| validation | 40 | 20 | 20 | Select training settings |
| calibration | 24 | 12 | 12 | Fit probability calibration |
| test | 40 | 20 | 20 | Final development measurement |

Training, validation, and calibration cover refund eligibility, urgency under explicit rules, duplicate reports, coding versus billing requests, and appointment scheduling. Test includes 20 cases from those families and 20 from permission and event-order families absent from training, validation, and calibration. All cases are individually authored and assigned stable source groups; no record or group is shared across splits. However, cases share phrasing, question forms, policies, and the same synthetic author. Related constructions across splits can make results optimistic. These tiny public splits cannot establish independent generalization, calibration quality, or production readiness.

Each JSONL record includes `schema_version`, `id`, `group_id`, `content`, `question`, `criteria` (null or an object with `yes` and `no` strings), a binary integer `label`, `label_type`, `task_family`, `source`, `provenance`, `license`, `rationale`, `review_status`, and `split`. Only content, question, and criteria are model inputs. Labels, rationales, identifiers, family names, and split names must not enter the model input. The explicit policy in the seed treats missing qualifying facts as no; do not generalize that convention into a universal rule for unknown answers.

Run `uv run data/build_seed.py` from the repository root to serialize the authored cases again without a network model call. This generator is the current authoritative source for this seed; edit it when correcting these records so regeneration does not erase fixes. Future community contributions should live in separately validated JSONL files rather than growing this development generator indefinitely.

New examples should give the content, question, explicit criteria where needed, correct label, and a short explanation. Publish only material you have the right to share. Keep related cases in one source group and one split; do not move a known test failure into training while continuing to report that test as untouched. Adding data is useful evidence, not a guarantee that retraining fixes the mistake. See [the contribution specification](../specs/open-training.md).

## Questions without criteria

`plain-questions.jsonl` adds 100 original synthetic records with `criteria: null`: 64 training, 16 validation, 8 calibration, and 12 test examples, each split evenly balanced. It covers refund requests, appointment requests, coding and billing questions, and duplicate reports. These examples ask about explicit message content rather than imposing an unstated refund or urgency policy. An unused item alone does not mean its owner has requested a refund.

Each of its 50 content passages has two different questions with different correct labels. The pair shares a `group_id` and stays in the same split. This makes the question necessary: classifying the content alone cannot answer both correctly. Some paired questions are related but not literal opposites. Question wording varies, but the same-author and small-sample limitations still apply; all records remain `synthetic_unreviewed` even though the author checked label consistency.

Run `uv run data/build_plain_questions.py` to regenerate this separate dataset from its authored source. The same CC0 intent above applies to these original examples and source. Combining both files yields 364 records: 224 training, 56 validation, 32 calibration, and 52 test. Keep dataset file lists explicit in training records; runs using only `seed.jsonl` did not train or evaluate on this addition.
