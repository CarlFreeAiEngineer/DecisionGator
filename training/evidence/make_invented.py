#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate invented, public training records for the evidence check (see specs/evidence-check.md).

One `claude` call per item makes an invented evidence text, a short context, about 10 labeled statements and up to
2 "flips" (the supporting line of a supported statement is changed or removed, so the same statement becomes unsupported).
Checks: a verbatim/paraphrase statement is kept only if its `support` quote occurs exactly in the evidence; a flip is kept
only if the quote is absent from the edited evidence. Honest-forgetting statements have no supporting line, so they carry no quote.
Output: data/evidence/invented.jsonl (plain, public, CC0-1.0). Resumable by item index (items already in the file are skipped).
    make_invented.py --items 12
"""
import argparse, json, random, re, shutil, subprocess, sys, threading, time, zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/evidence/invented.jsonl"
MODEL = "sonnet"
CHECK_MODEL = "haiku"
HELD_OUT_KIND = "game or sports recap"
KINDS = ["a person's notes or diary", "a chat between friends or coworkers", "a customer support thread",
         "meeting notes", "an interview transcript", "a news-style report", "a family history", "a project log",
         "a recipe story", "a travel journal", "a medical-intake style history (fictional)", "a company memo",
         HELD_OUT_KIND, "a character recalling their friendship with someone (friendly, not romantic)"]
SUBJECTS = ["a village bakery", "a river cleanup", "a school science fair", "a small software launch", "a lost dog",
            "a mountain hike", "a kitchen renovation", "a regional bike race", "a hospital move to a new building",
            "a neighborhood garden", "a delayed shipment", "a community choir", "a fishing trip", "a family reunion",
            "a bridge repair", "a robotics club", "a farmers market", "a night shift at a warehouse",
            "a chess tournament", "a street festival", "a broken heating system", "a tiny airline route",
            "a library renovation", "a cooking class", "a lighthouse volunteer program", "a used-car purchase",
            "a long train journey", "a pottery studio", "a beekeeping season", "a moving house"]
LENGTHS = [350, 600, 900, 1200, 1500, 1750]
FLIP_KINDS = ("name", "number", "date", "order", "speaker", "removed")
UNSUP = ("name", "number", "date", "order", "speaker", "added_detail", "never_told")
SUP = ("verbatim", "paraphrase", "forgetting")
KIND_OF = {"supported": SUP, "unsupported": UNSUP, "not_claim": ("question", "feeling_now", "happening_now", "plan")}

PROMPT = """You write invented training data for a program that checks whether a text supports a sentence. Everything must be fully invented and safe to publish: invented people, places and companies (no real people or brands), no romance, no sexual content, no graphic violence, no private or personal data, nothing political or spicy.

Make ONE item.
EVIDENCE kind: {kind}. Subject: {subject}. Length: about {words} words (never under 300, never over 1,800). Rich in invented names, numbers, dates, places and a clear order of events. Several different people or speakers. Spread the content evenly; do not front-load.
CONTEXT: a short reply or summary of 2 to 6 sentences that someone wrote using the evidence, in the voice that fits (a friend replying, an agent summarizing, a character recalling, a colleague reporting).
STATEMENTS: exactly 10 sentences to judge, each taken from the context or fitting it, readable in the context's voice. Write them in this exact order, each with exactly the kind given for its slot:
{slots}
Kinds:
- supported / verbatim: close to the evidence's own words. Give "support": one exact quote (6 to 30 words) copied character for character from the evidence.
- supported / paraphrase: quite different words, same content. Give "support" the same way.
- supported / forgetting: an honest "I don't remember" or "did you tell me?" about something the evidence does not cover. No support.
- unsupported / added_detail: one concrete, plausible detail (a tool, a place, a time of day, a remark) added to an otherwise true memory; the evidence never says it. No support.
- unsupported / never_told: a plausible fact on a topic the evidence never touches. No support.
- unsupported / name, number, date, order (two events swapped), speaker (who said or did it wrongly): subtle but real error, the evidence clearly contradicts or lacks it. No support.
Unsupported sentences should reuse evidence words so word overlap is no shortcut.
- not_claim: kind "question", "feeling_now" (a feeling now), "happening_now" (something happening now) or "plan". No support.
The supporting quotes of the three supported statements with quotes must come from different parts of the evidence: one from the first third, one from the middle, one from the last third.
FLIPS: pick 2 different supported statements that have a support quote (verbatim or paraphrase). For each, give "replacement": text that replaces its exact support quote in the evidence so the statement is no longer supported by anything in the evidence, and the fact is not stated elsewhere in the evidence. The kind of change is set for you:
{flip_spec}
(name: a different name; number: a different number; date: a different date or time; order: the events swapped; speaker: someone else said or did it; removed: replacement is "" or a short unrelated sentence.)

Return only JSON, no other text, in this shape:
{{"evidence": "...", "context": "...", "statements": [{{"text": "...", "label": "supported|unsupported|not_claim", "kind": "...", "support": "exact quote or empty"}}], "flips": [{{"statement": 1, "replacement": "..."}}, {{"statement": 2, "replacement": "..."}}]}}
"statement" in a flip is the 1-based number in the statements list; give the flips in the order listed above.{retry}"""

CHECK = """You label sentences for a training set. Below is an EVIDENCE text and numbered sentences. For each sentence decide whether the EVIDENCE supports it. Judge support, not truth.
- supported: the evidence holds what the sentence says about the past, in the same or other words. Honest forgetting or not knowing ("I don't remember", "did you tell me?") is supported.
- unsupported: it asserts a concrete, checkable fact about the past (event, name, number, time or order, place, who said or did what, any added detail) that the evidence does not hold or contradicts, even if it might be true.
- not_claim: a question, a feeling now, something happening now, a plan; nothing to check against the past.
Every part of a claim must be in the evidence.

=== EVIDENCE ===\n{evidence}\n=== END EVIDENCE ===

Sentences:\n{numbered}

Return only a JSON array of labels in order, for example ["supported", "unsupported"], no other text."""

STATS, LOCK = [], threading.Lock()


def claude_once(prompt, model=MODEL):
    t = time.time()
    out = subprocess.run(["claude", "-p", "--model", model, "--tools", "", "--setting-sources", "",
                          "--no-session-persistence", "--output-format", "json",
                          "--system-prompt", "You follow the instructions exactly."],
                         input=prompt, capture_output=True, text=True, timeout=900)
    if out.returncode != 0:
        raise RuntimeError("claude failed: " + out.stderr[-300:])
    j = json.loads(out.stdout)
    if j.get("is_error") or not (j.get("result") or "").strip():
        raise RuntimeError("claude returned no result: " + str(j.get("result"))[:200])
    u = j.get("usage", {})
    toks = sum(u.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    with LOCK:
        STATS.append((time.time() - t, toks, u.get("output_tokens", 0)))
    return j["result"]


def claude(prompt, model=MODEL):
    for attempt in range(3):
        try:
            return claude_once(prompt, model)
        except Exception as e:
            err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"claude failed 3 times: {err}")


def as_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text[text.find("{"):text.rfind("}") + 1])


def slots_for(seed, idx):
    r = random.Random(f"{seed}-slots-{idx}")
    nc = r.sample(KIND_OF["not_claim"], 3)
    slots = [("supported", "verbatim"), ("supported", "paraphrase"), ("supported", "paraphrase"),
             ("supported", "forgetting"), ("unsupported", "added_detail"), ("unsupported", "never_told"),
             ("unsupported", ("name", "number", "date", "order", "speaker")[idx % 5])] + [("not_claim", k) for k in nc]
    r.shuffle(slots)
    return slots


def second_check(ev, picks, seed, idx):
    """picks: list of (rec). Returns the set of ids where the checking model (haiku) disagrees."""
    numbered = "\n".join(f"{i + 1}. {r['statement']}" for i, r in enumerate(picks))
    got = as_list(claude(CHECK.format(evidence=ev, numbered=numbered), CHECK_MODEL))
    if len(got) != len(picks):
        raise RuntimeError("check answer has the wrong length")
    return {r["id"] for r, g in zip(picks, got) if g != r["label"]}


def as_list(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text[text.find("["):text.rfind("]") + 1])


def plan(seed, idx):
    r = random.Random(f"{seed}-{idx}")
    return {"kind": KINDS[idx % len(KINDS)], "subject": r.choice(SUBJECTS), "words": r.choice(LENGTHS),
            "split": split_for(seed, idx)}


def split_for(seed, idx):
    if KINDS[idx % len(KINDS)] == HELD_OUT_KIND:
        return "test_unseen"
    x = random.Random(f"{seed}-split-{idx}").random()
    return "train" if x < 0.80 else "validation" if x < 0.87 else "calibration" if x < 0.94 else "test"


def build(idx, seed, today):
    """Returns (records, dropped_reasons list). Raises if no usable item after retries."""
    p = plan(seed, idx)
    slots = slots_for(seed, idx)
    fk = [FLIP_KINDS[(idx * 2 + j) % 6] for j in range(2)]
    flip_spec = "\n".join(f"flip {j + 1}: {k}" for j, k in enumerate(fk))
    slot_text = "\n".join(f"{i}. {l} / {k}" for i, (l, k) in enumerate(slots, 1))
    gid = f"inv-{idx:05d}"
    bad = []
    for attempt in range(2):
        retry = "" if not bad else "\n\nYour previous answer had problems: " + "; ".join(bad[:5]) + ". Fix them."
        try:
            d = as_json(claude(PROMPT.format(slots=slot_text, flip_spec=flip_spec, kind=p["kind"], subject=p["subject"], words=p["words"], retry=retry)))
            ev, ctx, sts = d["evidence"].strip(), d["context"].strip(), d["statements"]
            assert isinstance(sts, list)
            got = [(x["label"], x["kind"]) for x in sts]
            if got != slots:
                bad = ["statements must follow the slot order and kinds exactly; wanted " + "; ".join(f"{l}/{k}" for l, k in slots)]
                continue
        except RuntimeError as e:
            if "claude failed" in str(e) or "no result" in str(e):
                raise
            bad = [f"unreadable answer ({str(e)[:80]})"]; continue
        except Exception as e:
            bad = [f"unreadable answer ({str(e)[:80]})"]; continue
        n = len(ev.split())
        if not 300 <= n <= 1800:
            bad = [f"evidence has {n} words, need 300 to 1800"]; continue
        break
    else:
        raise RuntimeError("no usable item: " + "; ".join(bad))
    prov = {"generator": f"{MODEL} via claude CLI", "date": today, "seed": seed, "item_index": idx}
    base = {"group_id": gid, "context": ctx, "source": "invented", "license": "CC0-1.0", "provenance": prov,
            "evidence_kind": p["kind"], "subject": p["subject"]}
    recs, dropped, kept = [], [], {}
    for i, s in enumerate(sts, 1):
        try:
            text, lab, kind = s["text"].strip(), s["label"], s["kind"]
            sup = (s.get("support") or "").strip()
        except Exception:
            dropped.append("malformed statement"); continue
        if not text or lab not in KIND_OF or kind not in KIND_OF[lab]:
            dropped.append("bad label or kind"); continue
        if lab == "supported" and kind != "forgetting":
            if not sup or sup not in ev:
                dropped.append("support quote not in evidence"); continue
        else:
            sup = ""
        pos = round(ev.find(sup) / max(1, len(ev)), 2) if sup else None
        rec = {"id": f"{gid}-{i:02d}", **base, "evidence": ev, "statement": text, "label": lab, "kind": kind,
               "split": p["split"], "support": sup, "support_position": pos}
        kept[i] = rec
        recs.append(rec)
    must = {"forgetting", "added_detail", "never_told"}
    others = [r for r in recs if r["kind"] not in must]
    picks = [r for r in recs if r["kind"] in must] + random.Random(f"{seed}-chk-{idx}").sample(others, min(2, len(others)))
    if picks:
        try:
            disagree = second_check(ev, picks, seed, idx)
        except Exception as e:
            raise RuntimeError(f"second check failed: {e}")
        for r in picks:
            if r["id"] in disagree:
                dropped.append("second check disagrees: " + r["kind"])
                recs.remove(r); kept = {k: v for k, v in kept.items() if v is not r}
    flips = []
    for j, f in enumerate((d.get("flips") or [])[:2]):
        try:
            r = kept.get(int(f["statement"]))
            rep = (f.get("replacement") or "")
        except Exception:
            dropped.append("malformed flip"); continue
        if not r or not r["support"] or r["id"] in {x["flip_of"] for x in flips}:
            dropped.append("flip of a statement without a kept quote"); continue
        if fk[j] != "removed" and not rep.strip():
            dropped.append("flip with empty replacement for kind " + fk[j]); continue
        ev2 = ev.replace(r["support"], rep, 1)
        if r["support"] in ev2 or ev2 == ev or not 250 <= len(ev2.split()) <= 1800:
            dropped.append("flip quote still present or edit unusable"); continue
        flips.append({**r, "id": r["id"] + "-flip", "evidence": ev2, "label": "unsupported", "kind": fk[j],
                      "support": "", "support_position": None, "flip_of": r["id"], "removed_quote": r["support"]})
    for fr in flips:  # the checker must also call each flipped statement unsupported on the edited evidence
        try:
            bad_flip = second_check(fr["evidence"], [fr], seed, idx)
        except Exception as e:
            raise RuntimeError(f"second check failed: {e}")
        if bad_flip:
            dropped.append("second check disagrees: flip " + fr["kind"])
        else:
            recs.append(fr)
    return recs, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=int, required=True, help="make items with index 0..N-1")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    if not shutil.which("claude"):
        sys.exit("error: the `claude` command is not on PATH. Install Claude Code and sign in first.")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.out.exists():
        for line in a.out.read_text("utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["provenance"]["item_index"])
    todo = [i for i in range(a.items) if i not in done]
    today = date.today().isoformat()
    print(f"{len(todo)} items to make ({len(done)} already in file), {a.workers} workers", file=sys.stderr)
    t0, ok, failed, dropped, new = time.time(), 0, [], Counter(), []
    with ThreadPoolExecutor(a.workers) as ex, a.out.open("a", encoding="utf-8") as f:
        futs = {ex.submit(build, i, a.seed, today): i for i in todo}
        for fu in as_completed(futs):
            i = futs[fu]
            try:
                recs, dr = fu.result()
            except Exception as e:
                print(f"failed item {i}: {e}", file=sys.stderr); failed.append(i); continue
            dropped.update(dr)
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            new += recs; ok += 1
            print(f"[{ok}/{len(todo)}] item {i}: {len(recs)} records, {len(dr)} dropped", file=sys.stderr)
    el = time.time() - t0
    print(f"\nSUMMARY: {ok} items, {len(new)} records, {len(failed)} failed {failed} in {el:.0f}s", file=sys.stderr)
    print(f"label {dict(Counter(r['label'] for r in new))}", file=sys.stderr)
    print(f"kind {dict(Counter(r['kind'] for r in new))}", file=sys.stderr)
    print(f"split {dict(Counter(r['split'] for r in new))}", file=sys.stderr)
    print(f"flips {sum(1 for r in new if 'flip_of' in r)}; dropped statements {sum(dropped.values())} {dict(dropped)}",
          file=sys.stderr)
    if STATS:
        print(f"calls {len(STATS)}: mean {sum(s[0] for s in STATS) / len(STATS):.0f}s, mean input tokens "
              f"{sum(s[1] for s in STATS) // len(STATS)}, mean output tokens {sum(s[2] for s in STATS) // len(STATS)}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
