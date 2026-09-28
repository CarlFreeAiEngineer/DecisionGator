# Multiple-choice routing with generic labels (choices-v2)

`choices-v2.jsonl` is a second original synthetic dataset for `choose(content, question, options)`, written to fix one measured weakness: when option labels are short and generic (for example "support", "billing", "sales"), the model treated a bare "support" option as a catch-all for anyone asking for help and treated any mention of money as sales. The records use the same schema as `choices.jsonl` (see `choices.md`), with `source` set to `decisiongator-original-choices-v2`, ids `choices-v2-001` to `choices-v2-160`, group ids `choices-v2-grp-001` to `choices-v2-grp-080`, and `provenance.created` set to 2026-09-28. Only `content`, `question`, and `options` are model inputs. Every record has `review_status: synthetic_unreviewed`: the author checked each label while writing, but nothing has had independent human review. The authors intend all records to be available under CC0-1.0 (public domain dedication), to the extent rights in generated material can be dedicated; see <https://creativecommons.org/publicdomain/zero/1.0/legalcode>.

There are three families. `support_routing` (100 records) varies the wording and number of options (2 to 6): "support", "customer support", "help desk", "tech support", "technical support", "engineering support", "billing", "payments", "finance", "sales", "account management", "shipping", "returns", "account access", with mixed capitalization and order. Its hard cases are billing problems phrased as pleas for help or with polite openers (refunds, duplicate or unexpected charges, bounced payments, invoices, receipts, tax registration details, card updates, charges after cancelling), pricing and purchase questions phrased as requests for help (sales, or account management for existing customers), technical faults that mention money or the plan (a checkout or card form that errors is technical, not billing), and messages with two topics where the question picks one. `request_type` (20 records) contrasts generic labels such as "Help", "General question", "Other" with specific ones such as "Refund", "Replacement", "Reschedule", "Complaint". `department_routing` (40 records) is a new family of internal workplace messages routed among HR, IT, facilities, payroll, finance, legal, "general enquiries" and similar, with traps such as a timesheet mentioned in a Wi-Fi problem or a contractor's unpaid invoice (accounts payable, not payroll).

Every group is one content passage under two questions, sharing one `group_id` and one `split`; about a quarter of groups have two different correct answers (two-topic messages or negated questions), and the rest ask the same routing decision with a different option set and question wording. The correct index is spread across positions. Splits are assigned by group number: group number modulo 20 equal to 0 or 1 is validation, equal to 2 is calibration, everything else is train. Three probe sentences used to measure the weakness (a polite request to refund a duplicate card charge, a failed payment where money still left the account, and a request for a receipt showing a VAT number) are deliberately kept out of the file, including close paraphrases, so they remain an independent check.

| Split | Records | Groups |
| --- | ---: | ---: |
| train | 136 | 68 |
| validation | 16 | 8 |
| calibration | 8 | 4 |
| total | 160 | 80 |

| Family | Records |
| --- | ---: |
| support_routing | 100 |
| department_routing | 40 |
| request_type | 20 |

## Held-out test (choices-v2-test)

`choices-v2-test.jsonl` holds 60 test records (40 `support_routing`, 20 `department_routing`, 43 groups), written by a separate author who never saw `choices-v2.jsonl`. Every record has `split: test`. Before the new data was added, version 0.4.1 answered 25 of 42 `support_routing` test cases correctly across this file and the `support_routing` records of `choices.jsonl`.

## Rebuilding

Both files are written by scripts that hold the hand-authored records: `uv run data/build_choices_v2.py data/choices-v2.jsonl` and `uv run data/build_choices_v2_test.py data/choices-v2-test.jsonl` reproduce them byte for byte.
