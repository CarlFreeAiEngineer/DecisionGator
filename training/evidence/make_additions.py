#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate invented, public "half-true" records for the evidence check (see specs/evidence-check.md).

The case this teaches: the evidence says Arnold has big muscles, the reply says "Arnold has large muscles and long
hair". The sentence keeps a true part and joins on ONE piece of new information that appears nowhere in the evidence.

One `claude` call per item makes an invented evidence text (a chat followed by an instruction, notes, a brief for a
writer, or a document; 150 to 1,800 words), a faithful reply of 4 to 7 sentences, 3 variants of supported sentences
with one new piece of information added (label unsupported, kind added_clause), and 1 variant that only rewords a
supported sentence (label supported, kind reworded), so rewording alone is no sign of invention. No wrong dates,
wrong order or changed numbers: only new information.

A second model (haiku) labels every sentence against the evidence; a variant or an original it disagrees with is dropped.

Records, in the schema of invented.jsonl: the reply's own sentences carry "sentence_index" and form one message.
Each variant is one more record with the same evidence, group, split and "context" (the original reply), plus
"variant_of": the id of the sentence record it replaces. The trainer builds the variant's message from the original
reply's sentences with that one sentence replaced.

Output: data/evidence/additions.jsonl (plain, gitignored) and additions.jsonl.gz (committed, CC0-1.0).
Resumable by item index.
    make_additions.py --items 700
"""
import argparse, gzip, json, random, shutil, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_invented import CHECK_MODEL, HELD_OUT_KIND, KINDS, MODEL, STATS, as_json, as_list, claude  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/evidence/additions.jsonl"
SHAPES = [
    "a chat history between two people (many short turns, each starting with the speaker's name), followed by a blank line and one line of instruction to the person who will write the next reply",
    "a person's own notes: uneven, dated or bulleted, written for themselves",
    "a brief for a writer: instructions on what to say plus the facts to use (\"Explain that ...\", \"Mention ...\")",
    "a document: {kind}",
]
MORE_KINDS = ["a product sheet", "a short biography of an invented person", "a how-to guide", "an incident report",
              "a club newsletter", "a school report", "a property listing", "a field-trip summary",
              "a maintenance manual page", "a grant application summary", "minutes of a tenants' meeting",
              "a museum exhibit label set", "a job handover note", "a wildlife survey log"]
SUBJECTS = ["a bodybuilder neighbor", "a second-hand bookshop", "a ferry timetable change", "a puppet theater",
            "a roof garden", "a village football coach", "an old tractor restoration", "a new espresso machine",
            "a marathon training plan", "a clockmaker's workshop", "a coastal weather station", "a school orchestra tour",
            "a food truck", "a canal boat holiday", "a flooded basement", "a quilting circle", "a telescope club",
            "a bakery apprentice", "a mountain rescue dog", "a printing press", "a tailor's shop", "a kayak rental",
            "a museum night guard", "an orchard harvest", "a board-game cafe", "a village postmaster",
            "a greenhouse experiment", "a bicycle repair stall", "an ice rink", "a retired sea captain",
            "a new puppy", "a pottery kiln failure", "a wedding cake order", "a school play", "a windmill restoration",
            "a taxi driver's long shift", "a small radio station", "a camping trip", "a shoe repair shop", "an allotment feud"]
LENGTHS = [150, 300, 500, 800, 1200, 1800]
POSITIONS = ("at the start of the sentence", "in the middle of the sentence", "at the end of the sentence")
JOINS = ("joined with \"and\"", "as a \"which\" or \"who\" clause", "as a relative clause or a participle phrase",
         "as an appositive between commas", "joined with \"with\" or \"while\"", "as a second short clause after a comma")
NEW_KINDS = ("a trait or appearance", "an object or possession", "an event that happened", "another person",
             "a place", "a remark someone made", "a habit", "a reason or cause")
NOT_CLAIM = ("question", "feeling_now", "happening_now", "plan")

PROMPT = """You write invented training data for a program that checks whether a text supports a sentence. Everything must be fully invented and safe to publish: invented people, places and companies (no real people or brands), no romance, no sexual content, no graphic violence, no private or personal data, nothing political or spicy.

Make ONE item.
EVIDENCE shape: {shape}. Subject: {subject}. Length: about {words} words (within 25 percent). Give it concrete invented names, traits, objects, places and events.
REPLY: {n} sentences that someone wrote using only the evidence, in the voice that fits (a friend replying, a writer following the brief, a colleague summarizing). Every sentence is faithful: either "supported" (everything it says is in the evidence, in the same or other words) or "not_claim" (a question, a feeling now, something happening now, or a plan; nothing to check). At least 4 sentences are supported. For each supported sentence give "support": one exact quote (4 to 30 words) copied character for character from the evidence. For each not_claim sentence give "kind": question, feeling_now, happening_now or plan.
ADDITIONS: pick 3 different supported sentences. For each, write a variant that keeps what the sentence says and joins on ONE clearly new piece of information that appears nowhere in the evidence and cannot be worked out from it. Like this: the evidence says only that Arnold has big muscles; the variant says "Arnold has large muscles and long hair". The new information must be concrete and plainly absent, not a rewording, not a natural consequence, not a vague flourish. Do NOT change a date, a number, a name or the order of events that the evidence gives: add, never alter. The kind of new information, where it sits and how it is joined are set for you:
{additions}
In "added" give just the new information in a few words.
REWORDED: pick one supported sentence (preferably not one of the three) and write a variant that says the same thing in clearly different words, or with a mild elaboration that adds no new fact. It must still be fully supported.

Return only JSON, no other text, in this shape:
{{"evidence": "...", "reply": [{{"text": "...", "label": "supported|not_claim", "support": "exact quote or empty", "kind": "empty or the not_claim kind"}}], "additions": [{{"sentence": 1, "text": "...", "added": "..."}}], "reworded": {{"sentence": 2, "text": "..."}}}}
"sentence" is the 1-based number of the reply sentence the variant replaces; give the additions in the order listed above.{retry}"""

CHECK = """You label sentences for a training set. Below is an EVIDENCE text and numbered sentences. For each sentence decide whether the EVIDENCE supports it. Judge support, not truth.
- supported: the evidence holds everything the sentence says, in the same or other words. Rewording and mild elaboration that adds no new fact are supported.
- unsupported: the sentence states some concrete information (a trait, an object, an event, a person, a place, a remark, a habit, a reason) that appears nowhere in the evidence and cannot be worked out from it. A sentence that mixes supported facts with ONE such piece of new information is unsupported.
- not_claim: a question, a feeling now, something happening now, a plan; nothing to check against the evidence.

=== EVIDENCE ===\n{evidence}\n=== END EVIDENCE ===

Sentences:\n{numbered}

Return only a JSON array of labels in order, for example ["supported", "unsupported"], no other text."""


def plan(seed, idx):
    r = random.Random(f"{seed}-add-{idx}")
    shape = SHAPES[idx % len(SHAPES)]
    if "{kind}" in shape:
        shape = shape.format(kind=r.choice(MORE_KINDS + KINDS))
    words = LENGTHS[(idx // len(SHAPES)) % len(LENGTHS)]
    if "brief for a writer" in shape and words > 800:
        words = r.choice(LENGTHS[:4])
    x = random.Random(f"{seed}-add-split-{idx}").random()
    split = "train" if x < 0.80 else "validation" if x < 0.87 else "calibration" if x < 0.93 else "test"
    if HELD_OUT_KIND in shape:  # make_invented.py keeps this kind out of training; so do we
        split = "test_unseen"
    specs = [(NEW_KINDS[(idx + 3 * j) % len(NEW_KINDS)], POSITIONS[(idx + j) % 3], JOINS[(idx * 5 + j * 2) % len(JOINS)])
             for j in range(3)]
    return {"shape": shape, "subject": r.choice(SUBJECTS), "words": words, "n": r.choice((4, 5, 5, 6, 6, 7)),
            "split": split, "specs": specs}


def build(idx, seed, today):
    """Returns (records, dropped reasons). Raises if no usable item after retries."""
    p = plan(seed, idx)
    additions = "\n".join(f"addition {j + 1}: {k}, {pos}, {join}" for j, (k, pos, join) in enumerate(p["specs"]))
    bad = []
    for attempt in range(3):
        retry = "" if not bad else "\n\nYour previous answer had problems: " + "; ".join(bad[:5]) + ". Fix them."
        try:
            d = as_json(claude(PROMPT.format(shape=p["shape"], subject=p["subject"], words=p["words"], n=p["n"],
                                             additions=additions, retry=retry)))
            ev = d["evidence"].strip()
            reply = [{"text": s["text"].strip(), "label": s["label"], "support": (s.get("support") or "").strip(),
                      "kind": (s.get("kind") or "").strip()} for s in d["reply"]]
            adds = [(int(a["sentence"]), a["text"].strip(), (a.get("added") or "").strip()) for a in d["additions"]]
            rew = (int(d["reworded"]["sentence"]), d["reworded"]["text"].strip())
        except RuntimeError as e:
            if "claude failed" in str(e) or "no result" in str(e):
                raise
            bad = [f"unreadable answer ({str(e)[:80]})"]; continue
        except Exception as e:
            bad = [f"unreadable answer ({str(e)[:80]})"]; continue
        bad, n = [], len(ev.split())
        if not 100 <= n <= 2200:
            bad.append(f"evidence has {n} words, need about {p['words']}")
        if not 4 <= len(reply) <= 7:
            bad.append("the reply needs 4 to 7 sentences")
        for i, s in enumerate(reply, 1):
            if s["label"] not in ("supported", "not_claim") or not s["text"]:
                bad.append(f"reply sentence {i}: label must be supported or not_claim")
            elif s["label"] == "supported" and (not s["support"] or s["support"] not in ev):
                bad.append(f"reply sentence {i}: the support quote is not an exact copy from the evidence")
        ok = lambda k: 1 <= k <= len(reply) and reply[k - 1]["label"] == "supported"
        if len(adds) != 3 or len({k for k, _, _ in adds}) != 3 or not all(ok(k) and t and a for k, t, a in adds):
            bad.append("need exactly 3 additions on 3 different supported sentences, each with text and added")
        elif any(t == reply[k - 1]["text"] for k, t, _ in adds):
            bad.append("an addition is identical to its sentence")
        if not ok(rew[0]) or not rew[1] or (ok(rew[0]) and rew[1] == reply[rew[0] - 1]["text"]):
            bad.append("reworded must replace a supported sentence with different words")
        if not bad:
            break
    else:
        raise RuntimeError("no usable item: " + "; ".join(bad[:3]))

    # The second model labels the originals and every variant.
    variants = [(k, t, "unsupported", "added_clause", a) for k, t, a in adds] + [(rew[0], rew[1], "supported", "reworded", "")]
    texts = [s["text"] for s in reply] + [v[1] for v in variants]
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    try:
        got = as_list(claude(CHECK.format(evidence=ev, numbered=numbered), CHECK_MODEL))
    except Exception as e:
        raise RuntimeError(f"second check failed: {e}")
    if len(got) != len(texts):
        raise RuntimeError("second check answer has the wrong length")
    dropped = []
    agreed = [g == s["label"] for g, s in zip(got, reply)]  # an original the checker disputes gets no record
    dropped += ["second check disagrees: original " + s["label"] for s, a in zip(reply, agreed) if not a]

    gid = f"add-{idx:05d}"
    prov = {"generator": f"{MODEL} via claude CLI", "checker": CHECK_MODEL, "date": today, "seed": seed, "item_index": idx}
    base = {"group_id": gid, "evidence": ev, "source": "invented", "license": "CC0-1.0", "provenance": prov,
            "evidence_kind": p["shape"], "subject": p["subject"], "split": p["split"]}

    def record(i, rid, context, text=None, **extra):
        s = reply[i - 1]
        sup = s["support"] if s["label"] == "supported" else ""
        kind = "paraphrase" if s["label"] == "supported" else (s["kind"] if s["kind"] in NOT_CLAIM else "other")
        return {"id": rid, **base, "context": context, "statement": text or s["text"], "label": s["label"], "kind": kind,
                "support": sup, "support_position": round(ev.find(sup) / max(1, len(ev)), 2) if sup else None,
                "sentence_index": i, **extra}

    whole = " ".join(s["text"] for s in reply)
    recs = [record(i, f"{gid}-{i:02d}", whole) for i in range(1, len(reply) + 1) if agreed[i - 1]]
    for v, ((k, text, label, kind, added), g) in enumerate(zip(variants, got[len(reply):]), 1):
        if not agreed[k - 1]:
            dropped.append(f"variant of a disputed original: {kind}"); continue
        if g != label:
            dropped.append(f"second check disagrees: {kind}"); continue
        rec = record(k, f"{gid}-{k:02d}-v{v}", whole, text, variant_of=f"{gid}-{k:02d}")
        rec.update(label=label, kind=kind, **({"added": added, "support": "", "support_position": None} if added else {}))
        recs.append(rec)
    if not any(r["kind"] == "added_clause" for r in recs):
        raise RuntimeError("no addition survived the second check")
    return recs, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=int, required=True, help="make items with index 0..N-1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261115)
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
    t0, ok, failed, dropped = time.time(), 0, [], Counter()
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
            ok += 1
            print(f"[{ok}/{len(todo)}] item {i}: {len(recs)} records, {len(dr)} dropped", file=sys.stderr)
    rows = [json.loads(x) for x in a.out.read_text("utf-8").splitlines() if x.strip()]
    rows.sort(key=lambda r: r["id"])
    with gzip.open(str(a.out) + ".gz", "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSUMMARY: {ok} items made, {len(failed)} failed {sorted(failed)} in {time.time() - t0:.0f}s", file=sys.stderr)
    print(f"whole file: {len({r['group_id'] for r in rows})} items, {len(rows)} records ({sum('variant_of' in r for r in rows)} are variants)", file=sys.stderr)
    print(f"label {dict(Counter(r['label'] for r in rows))}", file=sys.stderr)
    print(f"kind {dict(Counter(r['kind'] for r in rows))}", file=sys.stderr)
    print(f"split by item {dict(Counter(s for _, s in {(r['group_id'], r['split']) for r in rows}))}", file=sys.stderr)
    print(f"dropped this run {sum(dropped.values())} {dict(dropped)}", file=sys.stderr)
    if STATS:
        print(f"calls {len(STATS)}: mean {sum(s[0] for s in STATS) / len(STATS):.0f}s, mean input tokens "
              f"{sum(s[1] for s in STATS) // len(STATS)}, mean output tokens {sum(s[2] for s in STATS) // len(STATS)}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
