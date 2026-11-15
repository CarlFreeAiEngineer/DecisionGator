#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Real replies as unlabeled training records, for a small model to learn from a larger one (train_message.py --teacher).

    uv run training/evidence/unlabeled_replies.py app-hy3-x21 app-hy3-x22 ...

Splits each reply to a training question into sentences and writes one record per sentence with "unlabeled": true
(no call to any model). Replies to validation, calibration and test questions are left out.
Output: ../RealAiDating/evals/memory/evidence-labels/real-unlabeled.jsonl.gz (private; rewritten each time).
"""
import gzip, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.evidence.label_replies import MARK, OUT, RUNS, question_splits, split_sentences  # noqa: E402

heldout, n, out = question_splits(), 0, OUT.with_name("real-unlabeled.jsonl.gz")
with gzip.open(out, "wt", encoding="utf-8") as f:
    for name in sys.argv[1:]:
        d = RUNS / name
        model = re.fullmatch(r"app-([a-z0-9]+)-x?\d+", name).group(1)
        for q, rep in sorted(json.loads((d / "replies.json").read_text("utf-8")).items()):
            reply = (rep.get("reply") or "").strip()
            if not reply or q in heldout:
                continue
            evidence = (d / "prompts" / f"{q}.md").read_text("utf-8").split(MARK)[0].rstrip()
            for i, s in enumerate(split_sentences(reply)):
                f.write(json.dumps({"id": f"{name}-{q}-{i + 1}", "group_id": q, "evidence": evidence, "context": reply.strip('"'),
                                    "statement": s, "label": "not_claim", "unlabeled": True, "kind": "real", "split": "train",
                                    "model": model, "run": name, "question": q, "sentence_index": i + 1}, ensure_ascii=False) + "\n")
                n += 1
print(f"{n} unlabeled sentences from {len(sys.argv) - 1} runs")
