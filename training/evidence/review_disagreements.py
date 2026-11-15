#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Second review of the real test sentences where the model and the first label disagree.

    uv run training/evidence/review_disagreements.py runs/<name>/predictions.jsonl [--threshold 0.7]

Picks test sentences from real replies that are (a) labeled fine but scored at or above the threshold, or (b) labeled
unsupported but scored below it. A stronger model (opus) judges each one again without seeing either the first label
or the score. Output: ../RealAiDating/evals/memory/evidence-labels/test-reviewed.jsonl.gz (private; appended,
resumable; with --labeled-new give another --output, such as train-reviewed.jsonl.gz), in the shape --relabel reads: {"id", "label", "confidence", "type"}. Prints counts only.
"""
import argparse, gzip, json, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.evidence.make_invented import claude  # noqa: E402

DIR = Path(__file__).resolve().parents[2].parent / "RealAiDating/evals/memory/evidence-labels"
PROMPT = """A character wrote the REPLY below using only the EVIDENCE (everything the character had in mind). Judge ONE sentence of the reply.

Answer with one word:
- new: the sentence clearly states a concrete piece of information that appears nowhere in the evidence and does not follow from it (an invented event, person, name, object, place, remark or detail), even if the rest of the sentence is in the evidence.
- told_wrong: it retells something the evidence does hold, but with a wrong day, order, number, place or person.
- borderline: it only rewords, mildly elaborates, colours or reasonably infers from what the evidence holds, or you are unsure.
- fine: the evidence supports it, or it is not a claim about the past at all (a question, a feeling, a plan, something happening now, honest forgetting).

=== EVIDENCE ===
{evidence}
=== END EVIDENCE ===

Reply: {reply}

Sentence: {sentence}

One word: new, told_wrong, borderline or fine."""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("predictions", nargs="?")
    ap.add_argument("--labeled-new", action="store_true",
                    help="instead: review every real sentence outside the test split that the first label calls clearly new information")
    ap.add_argument("--test", action="store_true", help="with --labeled-new: the test split instead of the others")
    ap.add_argument("--borderline-too", action="store_true", help="with --labeled-new: also those the first label marked borderline")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--output", default=str(DIR / "test-reviewed.jsonl.gz"))
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    rows = {}
    for line in gzip.open(DIR / "real-replies.jsonl.gz", "rt", encoding="utf-8"):
        r = json.loads(line)
        rows[r["id"]] = r
    out = Path(a.output)
    done = {json.loads(l)["id"] for l in gzip.open(out, "rt", encoding="utf-8")} if out.exists() else set()
    todo = []
    if a.labeled_new:
        todo = [{"id": r["id"], "label": "unsupported"} for r in rows.values() if (r["split"] == "test") == a.test and r["id"] not in done
                and r["label"] == "unsupported" and r.get("type") == "new" and (a.borderline_too or not r.get("borderline"))]
    for line in ([] if a.labeled_new else open(a.predictions, encoding="utf-8")):
        p = json.loads(line)
        if p["split"] != "test" or p["kind"] != "real" or p["id"] in done or p["id"] not in rows:
            continue
        if (p["label"] == "unsupported") != (p["probs"][1] >= a.threshold):
            todo.append(p)
    print(f"{len(todo)} sentences to review ({sum(p['label'] == 'unsupported' for p in todo)} labeled unsupported)", flush=True)

    def one(p):
        r = rows[p["id"]]
        got = claude(PROMPT.format(evidence=r["evidence"], reply=r["context"], sentence=r["statement"]), "opus").strip().lower()
        verdict = next((v for v in ("told_wrong", "borderline", "new", "fine") if v in got), "borderline")
        label = "unsupported" if verdict in ("new", "told_wrong") else r["label"] if verdict == "borderline" else \
            (r["label"] if r["label"] != "unsupported" else "supported")
        return p, {"id": p["id"], "label": label, "confidence": "unsure" if verdict == "borderline" else "clear",
                   "type": verdict if verdict in ("new", "told_wrong") else "", "verdict": verdict, "previous_label": p["label"]}

    counts = Counter()
    with ThreadPoolExecutor(a.workers) as ex, gzip.open(out, "at", encoding="utf-8") as f:
        for p, rec in ex.map(one, todo):
            f.write(json.dumps(rec) + "\n"); f.flush()
            counts[("labeled unsupported, scored low" if p["label"] == "unsupported" else "labeled fine, scored high", rec["verdict"])] += 1
    for k, v in sorted(counts.items()):
        print(k, v)


if __name__ == "__main__":
    main()
