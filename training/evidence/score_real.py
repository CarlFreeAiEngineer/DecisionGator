#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Score a run's predictions.jsonl on the real test replies, by the model that wrote them.

    uv run training/evidence/score_real.py runs/<name>/predictions.jsonl [reviewed.jsonl.gz ...]

Reviewed files ({"id", "label", "confidence", "type"}, as --relabel reads) replace labels by id; a sentence the
reviewer was unsure about, or called told wrong, is left out.

For each threshold t (alarm when p(unsupported) >= t): the share of unsupported sentences caught, and the share of
messages with no unsupported sentence that raise an alarm. Sentences the trainer ignored (borderline, told wrong) are
not in the file, so they count neither way. Also the share of joined-on additions caught (kind added_clause).
Prints numbers only.
"""
import gzip, json, re, sys
from collections import defaultdict

THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
reviewed = {}
for path in sys.argv[2:]:
    for line in (gzip.open if path.endswith(".gz") else open)(path, "rt", encoding="utf-8"):
        item = json.loads(line)
        reviewed[item["id"]] = item
test = []
for r in rows:
    item = reviewed.get(r["id"])
    if item:
        if item.get("confidence") == "unsure" or (item["label"] == "unsupported" and item.get("type") == "told_wrong"):
            continue
        r["label"] = item["label"]
    if r["split"] == "test":
        test.append(r)


def family(r):
    m = re.match(r"app-([a-z0-9]+)-", r.get("run") or "")
    name = m.group(1) if m else "?"
    return "hy3 first run (reviewed)" if r.get("run") == "app-hy3-1" else "claude" if name in ("haiku", "sonnet") else name


groups = defaultdict(lambda: defaultdict(list))
for r in test:
    if r["kind"] == "real":
        for fam in (family(r), "all real"):
            groups[fam][(r["run"], r["question"])].append(r)
print("| Replies by | Inventions | Clean messages | " + " | ".join(f"caught / flagged at {t}" for t in THRESHOLDS) + " |")
print("| --- | ---: | ---: | " + " | ".join("---:" for _ in THRESHOLDS) + " |")
for fam, msgs in sorted(groups.items()):
    bad = [r["probs"][1] for m in msgs.values() for r in m if r["label"] == "unsupported"]
    clean = [m for m in msgs.values() if not any(r["label"] == "unsupported" for r in m)]
    cells = []
    for t in THRESHOLDS:
        caught = f"{sum(p >= t for p in bad) / len(bad):.0%}" if bad else "-"
        flagged = f"{sum(any(r['probs'][1] >= t for r in m) for m in clean) / len(clean):.1%}" if clean else "-"
        cells.append(f"{caught} / {flagged}")
    print(f"| {fam} | {len(bad)} | {len(clean)} | " + " | ".join(cells) + " |")
for kind in sorted({r["kind"] for r in test if r["label"] == "unsupported" and r["kind"] != "real"}):
    ps = [r["probs"][1] for r in test if r["kind"] == kind and r["label"] == "unsupported"]
    print(f"{kind} ({len(ps)} test sentences): caught " + ", ".join(f"{sum(p >= t for p in ps) / len(ps):.0%} at {t}" for t in THRESHOLDS))
