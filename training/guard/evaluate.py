#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["onnxruntime==1.22.1", "numpy", "tokenizers==0.21.4"]
# ///
"""Score a model bundle on the guard check: whole messages, judged as examples/guard_check.py judges them.

    uv run training/guard/evaluate.py score --bundle released/macos-arm64 --name 0.4.1
    uv run training/guard/evaluate.py report 0.4.1 0.4.3

`score` asks both questions of every message in data/local/guard/{validation,test,regression}.jsonl (a long
message in overlapping pieces, the highest answer counting) and saves the probabilities to
runs/guard/<name>[-criteria].json. `report` picks the two thresholds on the validation messages only (half of
the lowest answer given to a message that must be blocked) and counts misses and false alarms on the test.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'examples'))
import guard_check  # noqa: E402

DATA = ROOT / 'data/local/guard'
RUNS = ROOT / 'runs/guard'
SETS = ('validation', 'test', 'regression')


def scorer(bundle):
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    tokenizer = Tokenizer.from_file(str(bundle / 'tokenizer.json'))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    session = ort.InferenceSession(str(bundle / 'model.onnx'), options, providers=['CPUExecutionProvider'])
    temperature = manifest.get('temperature', 1.0)
    state = {'calls': 0, 'seconds': 0.0, 'longest': 0}

    def score(content, question, criteria):
        prompt = question if criteria is None else question + '\nYes: ' + criteria['yes'] + '\nNo: ' + criteria['no']
        encoded = tokenizer.encode(content, prompt)
        state['longest'] = max(state['longest'], len(encoded.ids))
        if len(encoded.ids) > manifest['max_tokens']:
            raise ValueError(f'{len(encoded.ids)} tokens')
        start = time.perf_counter()
        z = session.run(None, {k: np.array([v], dtype=np.int64) for k, v in
                               [('input_ids', encoded.ids), ('attention_mask', encoded.attention_mask), ('token_type_ids', encoded.type_ids)]})[0][0]
        state['calls'] += 1
        state['seconds'] += time.perf_counter() - start
        return float(1 / (1 + np.exp(-(float(z[1]) - float(z[0])) / temperature)))
    return score, state


def run_score(args):
    score, state = scorer(args.bundle)
    criteria = guard_check.CHILD_CRITERIA if args.criteria else None
    result = {'bundle': str(args.bundle), 'criteria': bool(args.criteria), 'sets': {}}
    for name in SETS:
        path = DATA / f'{name}.jsonl'
        if not path.is_file():
            continue
        rows = []
        for item in map(json.loads, path.read_text(encoding='utf-8').splitlines()):
            answer = guard_check.check(item['text'], item.get('known') or (), item.get('before') or '', score=score, criteria=criteria, always_both=True)
            rows.append({'id': item['id'], 'kind': item['kind'], 'want_sexual': item['sexual'], 'want_child': item['child'],
                         'sexual': answer['sexual'], 'child': answer['child'], 'pieces': answer['calls'] // 2, 'chars': len(item['text'])})
        result['sets'][name] = rows
        print(name, len(rows), flush=True)
    result.update(calls=state['calls'], ms_per_call=state['seconds'] / state['calls'] * 1000, longest_input_tokens=state['longest'])
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f'{args.name}{"-criteria" if args.criteria else ""}.json'
    out.write_text(json.dumps(result, indent=1) + '\n')
    print(f'{out}: {state["calls"]} calls, {result["ms_per_call"]:.0f} ms each in this runtime, longest input {state["longest"]} tokens')


def counts(rows, ts, tc):
    """Misses (should block, passed) and false alarms (should pass, blocked) at two thresholds."""
    block = [r for r in rows if r['want_sexual'] and r['want_child']]
    passes = [r for r in rows if not (r['want_sexual'] and r['want_child'])]
    blocked = lambda r: r['sexual'] >= ts and r['child'] >= tc
    return [r for r in block if not blocked(r)], [r for r in passes if blocked(r)], len(block), len(passes)


def choose(rows):
    """On validation messages only: for each question, half of the lowest answer given to a message that must be
    blocked, kept between 0.02 and 0.5. So the weakest such message still blocks with a margin of two."""
    block = [r for r in rows if r['want_sexual'] and r['want_child']]
    return tuple(min(.5, max(.02, round(min(r[q] for r in block) / 2, 2))) for q in ('sexual', 'child'))


def run_report(args):
    for name in args.names:
        result = json.loads((RUNS / f'{name}.json').read_text())
        sets = result['sets']
        ts, tc = (args.sexual, args.child) if args.sexual else choose(sets['validation'])
        print(f'\n## {name}: thresholds sexual {ts:g}, child {tc:g} ({"given" if args.sexual else "chosen on validation"})')
        for set_name in SETS:
            if set_name not in sets:
                continue
            rows = sets[set_name]
            misses, alarms, n_block, n_pass = counts(rows, ts, tc)
            print(f'{set_name}: misses {len(misses)} of {n_block} to block; false alarms {len(alarms)} of {n_pass} to pass')
            if set_name != 'validation':
                kinds = {}
                for r in rows:
                    k = kinds.setdefault(r['kind'], [0, 0])
                    k[0] += 1
                    k[1] += r in misses or r in alarms
                print('  wrong by kind: ' + ', '.join(f'{k} {v[1]}/{v[0]}' for k, v in sorted(kinds.items())))
                sexual_wrong = sum((r['sexual'] >= ts) != bool(r['want_sexual']) for r in rows)
                child_wrong = sum((r['child'] >= tc) != bool(r['want_child']) for r in rows)
                print(f'  each question alone: sexual wrong {sexual_wrong} of {len(rows)}, child wrong {child_wrong} of {len(rows)}')
            if args.show and set_name != 'validation':
                for label, group in (('MISS', misses), ('ALARM', alarms)):
                    for r in group[:args.show]:
                        print(f'  {label} {r["id"]} {r["kind"]} sexual {r["sexual"]:.2f} child {r["child"]:.2f}')
        if args.table and 'test' in sets:
            print('test at other thresholds (sexual, child: misses, false alarms):')
            for t in (.02, .05, .1, .2, .3, .5):
                misses, alarms, _, _ = counts(sets['test'], t, t)
                print(f'  {t:g}, {t:g}: {len(misses)}, {len(alarms)}')
        print(f'  {result["ms_per_call"]:.0f} ms per call in the scoring runtime; longest input {result["longest_input_tokens"]} tokens')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('score')
    s.add_argument('--bundle', required=True, type=Path)
    s.add_argument('--name', required=True)
    s.add_argument('--criteria', action='store_true', help='ask the child question with its criteria')
    r = sub.add_parser('report')
    r.add_argument('names', nargs='+')
    r.add_argument('--sexual', type=float)
    r.add_argument('--child', type=float)
    r.add_argument('--show', type=int, default=0, help='list this many wrong answers of each kind, by id')
    r.add_argument('--table', action='store_true')
    args = parser.parse_args()
    run_score(args) if args.command == 'score' else run_report(args)


if __name__ == '__main__':
    main()
