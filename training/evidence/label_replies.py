#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Turn real model replies (RealAiDating memory test) into sentence-level evidence-check records.

See specs/evidence-check.md. One `claude` call per reply labels every sentence. Output: data/evidence/real-replies.jsonl
Resumable: replies already in the output are skipped. Example pilot:
    label_replies.py --runs app-hy3-1 --limit 12 --verdicts wrong,partly
"""
import argparse, gzip, json, os, random, re, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ.get("REALAI_RUNS", ROOT.parent / "RealAiDating/evals/memory/runs"))
# Private: the labeled sentences hold RealAiDating's chat, so they live in that private repository, never here.
OUT = ROOT.parent / "RealAiDating/evals/memory/evidence-labels/real-replies.jsonl.gz"
LABEL_MODEL = "sonnet"
MARK = "# What to write"
LABELS = ("supported", "unsupported", "not_claim")

PROMPT = """You label sentences of a reply for a training set. A character wrote the REPLY using only the EVIDENCE below (what the character had in mind). Decide, for each numbered sentence, whether the EVIDENCE supports it. Judge support, not truth.

Labels:
- supported: the evidence holds what the sentence says about the past, in the same or other words. Honest forgetting or not knowing ("I have forgotten", "did you tell me?") is supported. Figurative or characterizing colour on a supported fact is supported too ("a beagle who thought he owned the bed" when the evidence says he slept on the bed).
- unsupported: the sentence asserts a concrete, checkable fact about the past (an event, a name, a number, a time or order, a place, who said or did what) that the evidence does not hold or contradicts, even if it might be true. This includes an invented concrete detail added to a true memory ("cursing the drill") and invented events in the speaker's own past ("the rain kept me busy"). When a sentence mixes supported facts with one invented fact, it is unsupported, and "why" names only the invented part.
- not_claim: questions, feelings or attitudes of the speaker then or now ("I did not like it"; if the rest of such a sentence is a supported fact, label it supported), things happening now, plans, greetings, opinions; nothing to check against the past.

A claim is supported only if every part of it is in the evidence. Check names, numbers, timing and sequence closely.

A judge who compared the reply with the whole chat gave this verdict (yes, partly, no, wrong): {verdict}. Reason: {why}
Use it only as a hint. We care only about what the EVIDENCE shows; the judge saw more.

=== EVIDENCE ===
{evidence}
=== END EVIDENCE ===

=== WHOLE REPLY ===
{reply}
=== END REPLY ===

Numbered sentences:
{numbered}

Return only a JSON array with one object per sentence, in order, no other text:
[{{"n": 1, "label": "supported|unsupported|not_claim", "support": ["exact quote copied character for character from the EVIDENCE"], "why": "one short line", "borderline": false}}]
"support" must be [] unless the label is supported (for supported claims, give one to three short exact quotes; for honest forgetting or when nothing applies, []).
"borderline" is true only when a careful person could reasonably choose another label.{retry}"""


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if len(text) >= 2 and text[0] in "\"'“" and text[-1] in "\"'”":
        text = text[1:-1].strip()
    out = []
    for para in re.split(r"\n+", text):
        for s in re.split(r"(?<=[.!?…])[\"'”)]*\s+(?=[\"'“(]?[A-Z0-9])", para.strip()):
            if s.strip():
                out.append(s.strip())
    return out


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def as_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("[")
    return json.loads(text[start:text.rfind("]") + 1])


STATS = []
LOCK = threading.Lock()


def claude(prompt: str) -> str:
    for attempt in range(3):
        try:
            return claude_once(prompt)
        except Exception as e:
            err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"claude failed 3 times: {err}")


def claude_once(prompt: str) -> str:
    t = time.time()
    out = subprocess.run(["claude", "-p", "--model", LABEL_MODEL, "--tools", "", "--setting-sources", "",
                          "--no-session-persistence", "--output-format", "json",
                          "--system-prompt", "You follow the instructions exactly."],
                         input=prompt, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError("claude failed: " + out.stderr[-300:])
    j = json.loads(out.stdout)
    if j.get("is_error") or not (j.get("result") or "").strip():
        raise RuntimeError("claude returned no result: " + str(j.get("result"))[:200])
    u = j.get("usage", {})
    toks = sum(u.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    with LOCK:
        STATS.append((time.time() - t, toks, u.get("output_tokens", 0)))
    return j.get("result", "")


def label(evidence, reply, sents, verdict, why):
    """Returns (items, needs_review). Retries once if a quote is not in the evidence or the shape is wrong."""
    ev = norm(evidence)
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sents))
    items, bad = None, ["not run"]
    for attempt in range(2):
        retry = "" if attempt == 0 else (
            "\n\nYour previous answer had problems: " + "; ".join(bad[:5]) +
            ". Quote the evidence exactly, or leave support empty and change the label.")
        try:
            got = as_json(claude(PROMPT.format(evidence=evidence, reply=reply, numbered=numbered, verdict=verdict,
                                               why=why, retry=retry)))
        except RuntimeError as e:
            if "claude failed" in str(e) or "no result" in str(e):
                raise
            bad = [f"unreadable answer ({str(e)[:80]})"]
            continue
        except Exception as e:
            bad = [f"unreadable answer ({str(e)[:80]})"]
            continue
        bad = []
        if not isinstance(got, list) or len(got) != len(sents):
            bad.append(f"need exactly {len(sents)} objects")
            continue
        for i, g in enumerate(got):
            if g.get("label") not in LABELS:
                bad.append(f"sentence {i + 1}: bad label")
            sup = g.get("support") or []
            if g.get("label") != "supported":
                g["support"] = []
            elif not isinstance(sup, list) or any(norm(q) not in ev for q in sup):
                bad.append(f"sentence {i + 1}: a quote is not in the evidence")
        items = got
        if not bad:
            return items, False
    if items is None or len(items) != len(sents):
        raise RuntimeError("no usable labels after retry: " + "; ".join(bad[:3]))
    return items, True


def load_reply(run_dir: Path, q: str, rep: dict, verd: dict, split: str, model: str, today: str):
    evidence = (run_dir / "prompts" / f"{q}.md").read_text("utf-8").split(MARK)[0].rstrip()
    reply = rep["reply"]
    sents = split_sentences(reply)
    items, review = label(evidence, reply, sents, verd.get("verdict", "?"), verd.get("why", ""))
    recs = []
    for i, s in enumerate(sents):
        g = items[i] if isinstance(items[i], dict) else {}
        lab = g.get("label") if g.get("label") in LABELS else "not_claim"
        recs.append({
            "id": f"{run_dir.name}-{q}-{i + 1}", "group_id": q, "evidence": evidence, "context": reply.strip().strip('"'),
            "statement": s, "label": lab, "kind": "real", "source": f"RealAiDating/evals/memory/runs/{run_dir.name}",
            "split": split, "license": "CC0-1.0",
            "provenance": {"generated_by": f"labeling model {LABEL_MODEL} via claude CLI", "date": today,
                           "note": "labeled against the prompt evidence; judge verdict used only as a hint"},
            "support": g.get("support") or [], "why": g.get("why", ""), "borderline": bool(g.get("borderline", False)), "model": model, "run": run_dir.name,
            "question": q, "sentence_index": i + 1, "judge_verdict": verd.get("verdict", "?"),
            **({"needs_review": True} if review else {})})
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", help="run folder names (default: every app-<model>-<n>)")
    ap.add_argument("--limit", type=int, help="label at most this many new replies")
    ap.add_argument("--verdicts", help="only replies with these judge verdicts, comma separated (e.g. wrong,partly)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    if not shutil.which("claude"):
        sys.exit("error: the `claude` command is not on PATH. Install Claude Code and sign in first.")
    if not RUNS.is_dir():
        sys.exit(f"error: run folders not found at {RUNS} (set REALAI_RUNS)")
    names = a.runs or sorted(p.name for p in RUNS.iterdir() if re.fullmatch(r"app-[a-z0-9]+-\d+", p.name))
    allq = sorted({f"P{n:03d}" for n in range(1, 107)})
    pick = random.Random(20261114).sample(allq, 20)
    heldout = {q: ("validation" if i < 10 else "calibration") for i, q in enumerate(pick)}
    want = set(a.verdicts.split(",")) if a.verdicts else None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.out.exists():
        for line in gzip.open(a.out, "rt", encoding="utf-8"):
            r = json.loads(line)
            done.add((r["run"], r["question"]))
    jobs = []
    for name in names:
        d = RUNS / name
        m = re.fullmatch(r"app-([a-z0-9]+)-\d+", name)
        if not m or not (d / "replies.json").exists():
            print(f"skip {name}: not an app run with replies.json", file=sys.stderr)
            continue
        model = m.group(1)
        reps = json.loads((d / "replies.json").read_text("utf-8"))
        verds = json.loads((d / "verdicts.json").read_text("utf-8"))
        for q in sorted(reps):
            if not (reps[q].get("reply") or "").strip() or (name, q) in done or q not in verds:
                continue
            if want and verds[q].get("verdict") not in want:
                continue
            split = "test" if model == "hy3" else heldout.get(q, "train")
            jobs.append((d, q, reps[q], verds[q], split, model))
    if a.limit:
        jobs = jobs[:a.limit]
    today = date.today().isoformat()
    print(f"{len(jobs)} replies to label with {a.workers} workers", file=sys.stderr)
    n, failed, t0, recs_all = 0, [], time.time(), []
    with ThreadPoolExecutor(a.workers) as ex, gzip.open(a.out, "at", encoding="utf-8") as f:
        futs = {ex.submit(load_reply, *j[:4], j[4], j[5], today): j for j in jobs}
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                recs = fu.result()
            except Exception as e:
                print(f"failed {j[0].name} {j[1]}: {e}", file=sys.stderr)
                failed.append(f"{j[0].name} {j[1]}")
                continue
            recs_all += recs
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            print(f"[{n}/{len(jobs)}] {j[0].name} {j[1]}: {len(recs)} sentences", file=sys.stderr)
    from collections import Counter
    print(f"\nSUMMARY: {n} replies done, {len(failed)} skipped (failed) in {time.time() - t0:.0f}s", file=sys.stderr)
    if failed:
        print("skipped (rerun to retry): " + ", ".join(failed), file=sys.stderr)
    allr = [json.loads(x) for x in gzip.open(a.out, "rt", encoding="utf-8")] if a.out.exists() else []
    for title, rs in (("this run", recs_all), ("whole file", allr)):
        print(f"{title}: {len(rs)} records; label {dict(Counter(r['label'] for r in rs))}; "
              f"split {dict(Counter(r['split'] for r in rs))}; model {dict(Counter(r['model'] for r in rs))}; "
              f"needs_review {sum(1 for r in rs if r.get('needs_review'))}; "
              f"borderline {sum(1 for r in rs if r.get('borderline'))}", file=sys.stderr)
    if STATS:
        print(f"calls {len(STATS)}: mean {sum(s[0] for s in STATS) / len(STATS):.1f}s, "
              f"mean input tokens {sum(s[1] for s in STATS) // len(STATS)}, "
              f"mean output tokens {sum(s[2] for s in STATS) // len(STATS)}", file=sys.stderr)


if __name__ == "__main__":
    main()
