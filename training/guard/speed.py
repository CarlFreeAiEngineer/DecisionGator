#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Time the guard check through the Python package, as an application calls it.

    uv run training/guard/speed.py [BUNDLE]      # default: the bundle the package finds for this platform
"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'examples')]
import decisiongator  # noqa: E402
import guard_check  # noqa: E402

session = decisiongator.Session.load(sys.argv[1]) if len(sys.argv) > 1 else None
score = session.evaluate if session else decisiongator.is_yes_p
items = [json.loads(line) for line in (ROOT / 'data/local/guard/test.jsonl').read_text(encoding='utf-8').splitlines()]
score('warm up', guard_check.SEXUAL_QUESTION, None)
calls = []


def timed(content, question, criteria):
    start = time.perf_counter()
    p = score(content, question, criteria)
    calls.append((time.perf_counter() - start) * 1000)
    return p


for label, group in (('short messages (one piece)', [i for i in items if not i['kind'].startswith('long')][::8]),
                     ('long messages (130 to 220 words)', [i for i in items if i['kind'].startswith('long')][::3])):
    totals, n_calls = [], []
    calls.clear()
    for item in group:
        start = time.perf_counter()
        answer = guard_check.check(item['text'], item['known'], item['before'], score=timed)
        totals.append((time.perf_counter() - start) * 1000)
        n_calls.append(answer['calls'])
    print(f'{label}: {len(group)} messages; per call median {statistics.median(calls):.0f} ms, 95th percentile {sorted(calls)[int(len(calls) * .95)]:.0f} ms; '
          f'per message median {statistics.median(totals):.0f} ms, slowest {max(totals):.0f} ms; calls per message {statistics.mean(n_calls):.1f}')
