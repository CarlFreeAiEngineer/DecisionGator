#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Turn real `supported` sentences into `unsupported` ones by joining on one clearly new piece of information.

For each chosen sentence a sonnet call rewrites it in the same voice, keeping what it says and adding one concrete
thing the evidence never mentions (the "big muscles and long hair" shape). A haiku call must then judge the new
sentence unsupported. Each kept record has kind "added_clause" and variant_of = the original record's id, so the
per-message trainer forms the reply with that one sentence replaced. Test-split records get variants too, as a
second, larger test. Output: ../RealAiDating/evals/memory/evidence-labels/real-additions.jsonl.gz
(private; appended, resumable). Only counts are printed, never record text.
"""
import argparse, gzip, json, random, shutil, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.evidence.make_invented import claude  # noqa: E402
from training.evidence.flip_real import PROMPT as CHECK  # noqa: E402

DIR = Path(__file__).resolve().parents[2].parent / "RealAiDating/evals/memory/evidence-labels"
WRITE = """Below is the evidence a character wrote a reply from, the reply, and one sentence of it that the evidence supports.

Rewrite that one sentence so it also states ONE clearly new, concrete piece of information that appears nowhere in the evidence and does not follow from it: {what}. Keep everything the sentence already says, keep the character's voice and style, and join the new part naturally {where}. Do not change dates, numbers, names or the order of events that are already there. The result must still read as one sentence of the same reply.

Evidence:
{evidence}

Reply: {reply}

Sentence: {statement}

Answer with only the rewritten sentence."""
WHAT = ["a new trait or habit of a person or animal", "an object or possession never mentioned", "a past event that never happened in the evidence",
        "something another person supposedly said", "a place never mentioned", "a new person never mentioned",
        "a detail of how something was done", "something the speaker supposedly did"]
WHERE = ["at the end", "at the end", "in the middle", "at the start"]


def read(path):
    rows = []
    try:
        for line in gzip.open(path, "rt", encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    except EOFError:  # the labeler may still be appending
        pass
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DIR / "real-replies.jsonl.gz"))
    ap.add_argument("--output", default=str(DIR / "real-additions.jsonl.gz"))
    ap.add_argument("--per-split", default="train=1500,validation=150,calibration=150,test=500")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if not shutil.which("claude"):
        sys.exit("error: the `claude` command is not on PATH.")
    out, tried_path = Path(a.output), Path(a.output + ".tried")
    want = {k: int(v) for k, v in (x.split("=") for x in a.per_split.split(","))}
    pool = [r for r in read(a.input) if r["label"] == "supported" and not r.get("borderline") and len(r["statement"]) > 25]
    done = set(tried_path.read_text().split()) if tried_path.exists() else set()
    have = {}
    for r in (read(out) if out.exists() else []):
        have[r["split"]] = have.get(r["split"], 0) + 1
    rng = random.Random(a.seed)
    todo = []
    for split, n in want.items():
        by_model = {}
        for r in sorted((r for r in pool if r["split"] == split and r["id"] not in done), key=lambda r: r["id"]):
            by_model.setdefault(r.get("model"), []).append(r)
        for v in by_model.values():
            rng.shuffle(v)
        need = max(0, int((n - have.get(split, 0)) * 1.15))  # a little extra for the ones the check drops
        while need and any(by_model.values()):
            for m in sorted(by_model, key=str):  # take turns across models
                if by_model[m] and need:
                    todo.append(by_model[m].pop()); need -= 1
    print(f"candidates {len(pool)}, have {have}, to try {len(todo)}", flush=True)
    lock, counts = threading.Lock(), {"tried": 0, "kept": 0, "dropped": 0, "errors": 0}

    def work(r):
        pick = random.Random(r["id"])
        new = claude(WRITE.format(what=pick.choice(WHAT), where=pick.choice(WHERE), evidence=r["evidence"], reply=r["context"],
                                  statement=r["statement"])).strip().strip('"')
        ok = "\n" not in new and new != r["statement"] and len(new) < len(r["statement"]) + 200
        ok = ok and claude(CHECK.format(evidence=r["evidence"], statement=new), "haiku").strip().lower().startswith("unsupported")
        return r, new, ok

    with ThreadPoolExecutor(a.workers) as ex, gzip.open(out, "at", encoding="utf-8") as fo, open(tried_path, "a") as ft:
        for fut in as_completed([ex.submit(work, r) for r in todo]):
            try:
                r, new, ok = fut.result()
            except Exception:
                counts["errors"] += 1
                continue
            with lock:
                counts["tried"] += 1
                counts["kept" if ok else "dropped"] += 1
                if ok:
                    rec = dict(r, id=r["id"] + "-add", statement=new, label="unsupported", kind="added_clause", support=[],
                               why="", borderline=False, variant_of=r["id"])
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
                ft.write(r["id"] + "\n"); ft.flush()
                if counts["tried"] % 100 == 0:
                    print(counts, flush=True)
    print(counts)


if __name__ == "__main__":
    main()
