#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.21.4"]
# ///
"""Turn the checked texts of make_data.py into training records and a fixed test list.

    uv run training/guard/build_rows.py

Reads data/local/guard/items-train.jsonl and items-test.jsonl. Writes, beside them:
- train.jsonl: records in the pipeline's format (splits train, validation, calibration), for --extra-data.
- test.jsonl: the held-out messages (text, known, before, sexual, child), judged whole by evaluate.py.
- validation.jsonl: the validation messages in the same form, for choosing thresholds.

Background lines ("Known", "Before") are also added to texts written without any, so the model meets a
child who is in the background but not in the message (still no), and a child in the message under any
background (still yes). Nothing here is random between runs: every choice comes from a hash of the text.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'examples'))
from guard_check import CHILD_CRITERIA, CHILD_QUESTION, SEXUAL_QUESTION, child_content  # noqa: E402

DATA = ROOT / 'data/local/guard'
MAX_TOKENS = 256
PRONOUN = re.compile(r'\b(she|he|her|him|his|hers|they|them|their)\b', re.I)


def chance(text, salt):
    return int(hashlib.sha256((salt + text).encode()).hexdigest()[:8], 16) / 0xffffffff


def read(name):
    path = DATA / name
    seen, items = set(), []
    for line in path.read_text(encoding='utf-8').splitlines() if path.is_file() else []:
        item = json.loads(line)
        if item['id'] not in seen:
            seen.add(item['id'])
            items.append(item)
    return items


def with_background(items):
    """Give some texts written without background one borrowed from the pairs of the same set."""
    pools = {side: sorted({(i['known'], i['before']) for i in items if i['kind'].endswith('_' + side)}) for side in ('child', 'adult')}
    result = []
    for item in items:
        item = dict(item)
        if not item['kind'].startswith(('pair', 'long')) and pools['child'] and pools['adult']:
            roll = chance(item['text'], 'background')
            share_child, share_adult = (.30, .15) if not item['child'] else (.15, .10)
            side = 'child' if roll < share_child else 'adult' if roll < share_child + share_adult else None
            if side:
                known, before = pools[side][int(chance(item['text'], 'which') * len(pools[side])) % len(pools[side])]
                names = set(re.findall(r'\b[A-Z][a-z]+\b', known + ' ' + before))
                # A message without a child must not seem to point at the borrowed child: no shared name, no bare pronoun.
                clash = any(re.search(rf'\b{n}\b', item['text']) for n in names) or (side == 'child' and not item['child'] and PRONOUN.search(item['text']))
                if not clash:
                    item.update(known=known, before=before, borrowed=side)
        result.append(item)
    return result


def record(item, question, content, label, criteria, family, split, key):
    return {'schema_version': 1, 'id': f'{item["id"]}-{key}', 'group_id': item['group'], 'content': content, 'question': question,
            'criteria': criteria, 'label': int(label), 'label_type': 'binary', 'task_family': family, 'source': 'training/guard/make_data.py',
            'provenance': {'type': 'synthetic', 'generator': 'AI assistant', 'checked_by': 'AI assistant'}, 'license': 'unpublished',
            'rationale': f'Written as {item["kind"]}; a second model agreed.', 'review_status': 'synthetic_unreviewed', 'split': split}


def main():
    tokenizer = Tokenizer.from_file(str(ROOT / 'released/macos-arm64/tokenizer.json'))
    criteria_text = '\nYes: ' + CHILD_CRITERIA['yes'] + '\nNo: ' + CHILD_CRITERIA['no']

    def fits(content, question):
        return len(tokenizer.encode(content, question).ids) <= MAX_TOKENS

    train_items = with_background(read('items-train.jsonl'))
    test_items = with_background(read('items-test.jsonl'))
    test_texts = {i['text'].strip().casefold() for i in test_items}
    rows, seen, counts, validation, too_long = [], set(), {}, [], 0
    for item in train_items:
        if item['text'].strip().casefold() in test_texts:
            continue
        roll = chance(item['group'], 'split')
        split = 'train' if roll < .88 else 'validation' if roll < .95 else 'calibration'
        content = child_content(item['text'], item['known'], item['before'])
        if not fits(content, CHILD_QUESTION + criteria_text):
            too_long += 1
            continue
        if split == 'validation':
            validation.append(item)
        # The child question, half the time with its criteria. Validation and calibration get both forms.
        forms = [None, CHILD_CRITERIA] if split != 'train' else [CHILD_CRITERIA if chance(content, 'criteria') < .5 else None]
        for criteria in forms:
            if (content.strip().casefold(), bool(criteria)) not in seen:  # the same text can be written twice
                seen.add((content.strip().casefold(), bool(criteria)))
                rows.append(record(item, CHILD_QUESTION, content, item['child'], criteria, 'guard_child', split, 'cc' if criteria else 'c'))
        # The sexual question, on the bare message, once per text.
        if item['text'].strip().casefold() not in seen and (split != 'train' or chance(item['text'], 'sexual') < .6):
            seen.add(item['text'].strip().casefold())
            rows.append(record(item, SEXUAL_QUESTION, item['text'], item['sexual'], None, 'guard_sexual', split, 's'))
    for row in rows:
        key = (row['split'], row['task_family'], row['label'])
        counts[key] = counts.get(key, 0) + 1
    (DATA / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    keep = ('id', 'kind', 'text', 'known', 'before', 'sexual', 'child', 'borrowed', 'author')
    for name, items in (('test.jsonl', test_items), ('validation.jsonl', validation)):
        (DATA / name).write_text(''.join(json.dumps({k: i[k] for k in keep if k in i}, ensure_ascii=False) + '\n' for i in items), encoding='utf-8')
    print(f'train.jsonl: {len(rows)} records ({too_long} texts over {MAX_TOKENS} tokens left out)')
    for key in sorted(counts):
        print(' ', *key, counts[key])
    for name, items in (('test', test_items), ('validation', validation)):
        kinds = {}
        for i in items:
            kinds[i['kind']] = kinds.get(i['kind'], 0) + 1
        block = sum(1 for i in items if i['sexual'] and i['child'])
        print(f'{name}.jsonl: {len(items)} messages, {block} to block, {len(items) - block} to pass; by kind: {dict(sorted(kinds.items()))}')


if __name__ == '__main__':
    main()
