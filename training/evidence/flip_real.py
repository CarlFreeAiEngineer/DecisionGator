#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Flip real `supported` records into `unsupported` ones by deleting the supporting quotes from the evidence.

Only train/validation/calibration records are used, never test. A haiku call checks each flipped record and only
flips it judges unsupported are kept. Output: ../RealAiDating/evals/memory/evidence-labels/real-flips.jsonl.gz
(private; appended, resumable). Only counts are printed, never record text.
"""
import argparse, gzip, json, random, re, shutil, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.evidence.make_invented import claude  # noqa: E402

DIR = Path(__file__).resolve().parents[2].parent / "RealAiDating/evals/memory/evidence-labels"
SPLITS = ("train", "validation", "calibration")
PROMPT = """Below is some evidence and one sentence. Decide whether the evidence supports the sentence: the sentence's
facts are stated in the evidence or follow directly from it. A true fact that the evidence does not hold is NOT supported.

Evidence:
{evidence}

Sentence: {statement}

Answer with exactly one word: supported or unsupported."""


def remove_quotes(evidence, quotes):
    for q in quotes:
        if not isinstance(q, str) or not q.strip() or q not in evidence:
            return None
    for q in quotes:
        lines = evidence.split("\n")
        if any(l.strip() == q.strip() for l in lines):
            lines = [l for l in lines if l.strip() != q.strip()]
            evidence = "\n".join(lines)
        evidence = evidence.replace(q, "")
    return evidence


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DIR / "real-replies.jsonl.gz"))
    ap.add_argument("--output", default=str(DIR / "real-flips.jsonl.gz"))
    ap.add_argument("--target", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if not shutil.which("claude"):
        sys.exit("error: the `claude` command is not on PATH.")
    out, tried_path = Path(a.output), Path(a.output + ".tried")
    rows = [json.loads(l) for l in gzip.open(a.input, "rt", encoding="utf-8") if l.strip()]
    pool = [r for r in rows if r["label"] == "supported" and r.get("support") and r["split"] in SPLITS]
    missing, usable = 0, []
    for r in pool:
        ev = remove_quotes(r["evidence"], r["support"])
        if ev is None:
            missing += 1
        else:
            usable.append((r, ev))
    rng = random.Random(a.seed)
    by_model = {}
    for item in sorted(usable, key=lambda x: x[0]["id"]):
        by_model.setdefault(item[0].get("model"), []).append(item)
    for v in by_model.values():
        rng.shuffle(v)
    chosen, models = [], sorted(by_model, key=str)  # balanced: take turns across models
    while len(chosen) < a.target and any(by_model.values()):
        for m in models:
            if by_model[m] and len(chosen) < a.target:
                chosen.append(by_model[m].pop())
    done = set()
    if tried_path.exists():
        done = set(tried_path.read_text().split())
    todo = [(r, ev) for r, ev in chosen if r["id"] not in done]
    print(f"candidates {len(pool)}, missing quotes {missing}, chosen {len(chosen)}, already tried {len(chosen) - len(todo)}", flush=True)
    lock, counts = threading.Lock(), {"tried": 0, "kept": 0, "dropped": 0, "errors": 0}

    def work(item):
        r, ev = item
        ans = claude(PROMPT.format(evidence=ev, statement=r["statement"]), "haiku").strip().lower()
        return r, ev, ans.startswith("unsupported")

    with ThreadPoolExecutor(a.workers) as ex, gzip.open(out, "at", encoding="utf-8") as fo, open(tried_path, "a") as ft:
        for fut in as_completed([ex.submit(work, i) for i in todo]):
            try:
                r, ev, unsup = fut.result()
            except Exception:
                counts["errors"] += 1
                continue
            with lock:
                counts["tried"] += 1
                if unsup:
                    counts["kept"] += 1
                    new = dict(r, id=r["id"] + "-flip", evidence=ev, label="unsupported", kind="removed",
                               support=[], flipped_from=r["id"])
                    fo.write(json.dumps(new, ensure_ascii=False) + "\n"); fo.flush()
                else:
                    counts["dropped"] += 1
                ft.write(r["id"] + "\n"); ft.flush()
                if counts["tried"] % 100 == 0:
                    print(counts, flush=True)
    print(f"tried {counts['tried']}, kept {counts['kept']}, dropped by haiku {counts['dropped']}, "
          f"errors {counts['errors']}, missing quotes {missing}")


if __name__ == "__main__":
    main()
