"""Retrain DecisionGator with your corrections, using the same recipe as the released model.

    uv run training/retrain.py

1. Add your examples to data/corrections.jsonl (one per line; see WRONG-ANSWER.md).
2. Run this on a machine with an NVIDIA GPU of 40 GB or more (a paid Google Colab A100 works;
   colab/retrain.ipynb does everything for you). It takes about half an hour.
3. Unzip the result into your DecisionGator bundle folder, replacing model.onnx, tokenizer.json
   and manifest.json. The one-file DLL, NuGet package and Java JAR carry the model inside and
   must be rebuilt instead.

The script checks your corrections, trains from the original base model on all of the project's
training data plus your corrections (the 0.4.1 recipe), compresses and calibrates the model like
0.4.1, then scores it on every held-out test next to the published 0.4.1 scores and shows how the
old and new models answer each of your corrections.
"""
import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import pipeline  # noqa: E402

CORRECTIONS = ROOT / 'data/corrections.jsonl'
RELEASE_URL = 'https://62-84-178-253.sslip.io/DecisionGator/files/0.4.1/linux-x64/'

# The 0.4.1 recipe, from released/*/manifest.json and reports/accuracy-v4.md.
BASE = 'MoritzLaurer/deberta-v3-large-zeroshot-v2.0'
REVISION = 'cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2'
DEFAULT_DATA = [ROOT / 'data/seed.jsonl', ROOT / 'data/plain-questions.jsonl']  # pipeline.data_paths always adds these
TRAINING_DATA = [ROOT / 'data/expansion-v2.jsonl', ROOT / 'data/choices.jsonl',
                 *sorted(p for p in (ROOT / 'data/v4').glob('*.jsonl') if not p.name.startswith('test-'))]
# Held-out tests, scored like reports/accuracy-v4.1.md. None of these may be trained on.
TESTS = {
    'New test': [ROOT / f'data/v4/test-{i}.jsonl' for i in range(1, 5)],
    'Fresh test': [ROOT / 'data/evaluation-v2.jsonl'],
    'Original test': DEFAULT_DATA,
    'Choice test': [ROOT / 'data/choices.jsonl'],
    'Spam test': [ROOT / 'data/spam-comments-test.jsonl'],
    'Email test': [ROOT / 'data/email-address-test.jsonl'],
}
PUBLISHED = {'New test': 'v4test', 'Fresh test': 'fresh-test', 'Original test': 'old-test',
             'Choice test': 'choice-test', 'Spam test': 'spam-test', 'Email test': 'email-test'}


def read_corrections(path):
    """Turn the short correction lines into full training records, with readable errors."""
    rows = []
    if not path.is_file():
        return rows
    today = datetime.date.today().isoformat()
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        where = f'{path.relative_to(ROOT)} line {number}'
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f'{where}: not valid JSON ({e.msg}). Each line must look like '
                     '{"text": "...", "question": "...", "answer": "yes"}')
        unknown = set(item) - {'text', 'question', 'answer', 'options', 'criteria', 'why'}
        if unknown:
            sys.exit(f'{where}: unknown field(s) {sorted(unknown)}; allowed: text, question, answer, options, criteria, why')
        for key in ('text', 'question', 'answer'):
            if not isinstance(item.get(key), str) or not item[key].strip():
                sys.exit(f'{where}: "{key}" is required and must be text')
        options, answer = item.get('options'), item['answer'].strip()
        record = {'schema_version': 1, 'content': item['text'], 'question': item['question'],
                  'criteria': item.get('criteria'), 'task_family': 'correction', 'source': 'data/corrections.jsonl',
                  'provenance': {'type': 'contributed', 'generator': 'person', 'created': today},
                  'license': 'CC0-1.0', 'rationale': item.get('why') or 'Correction of a wrong answer.',
                  'review_status': 'unreviewed', 'split': 'train'}
        if options is None:
            if answer.lower() not in ('yes', 'no'):
                sys.exit(f'{where}: "answer" must be "yes" or "no" (or give "options" and name one of them)')
            record.update(label_type='binary', label=int(answer.lower() == 'yes'))
        else:
            if not isinstance(options, list) or len(options) < 2 or not all(isinstance(o, str) and o.strip() for o in options):
                sys.exit(f'{where}: "options" must be a list of at least two texts')
            if answer not in options:
                sys.exit(f'{where}: "answer" must be exactly one of the options: {options}')
            record.update(label_type='choice', options=options, label=options.index(answer))
        key = hashlib.sha256(json.dumps([record['content'], record['question'], options], ensure_ascii=False).encode()).hexdigest()[:12]
        record['id'] = record['group_id'] = f'correction-{key}'
        rows.append(record)
    return rows


def input_keys(rows):
    return {(c['content'].strip().casefold(), pipeline.prompt(c).strip().casefold()) for c in pipeline.expand(rows)}


def check_against_tests(corrections):
    """A correction that copies a test case would make that test's score meaningless."""
    test_rows = [r for files in TESTS.values() for r in pipeline.read_data(files) if r['split'] == 'test']
    test_inputs = input_keys(test_rows)
    test_texts = {r['content'].strip().casefold() for r in test_rows}
    for row in corrections:
        if input_keys([row]) & test_inputs:
            sys.exit(f'Correction "{row["content"][:60]}" with that question is a held-out test case. '
                     'Training on it would make the test scores meaningless; reword it or remove it.')
        if row['content'].strip().casefold() in test_texts:
            print(f'Note: the text "{row["content"][:60]}" also appears in a held-out test with another question.')


def fetch_baseline(directory):
    """Download the released 0.4.1 model to compare against (about 600 MB, checked against the release manifest)."""
    if (directory / 'manifest.json').is_file():
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    expected = json.loads((ROOT / 'released/linux-x64/manifest.json').read_text())['sha256']
    for name in ('manifest.json', 'tokenizer.json', 'model.onnx'):
        print(f'Downloading released 0.4.1 {name} ...', flush=True)
        urllib.request.urlretrieve(RELEASE_URL + name, directory / name)
        if name in expected and pipeline.digest(directory / name) != expected[name]:
            sys.exit(f'Downloaded {name} does not match the release checksum')
    return directory


def score(bundle, files, split, output):
    pipeline.evaluate(SimpleNamespace(command='evaluate', bundle=str(bundle), data=[str(f) for f in files], extra_data=[],
                                      split=split, output=str(output)))
    return json.loads(Path(output).read_text())


def percent(result):
    value = result.get('accuracy') if result.get('count') else result.get('choice_accuracy')
    return f'{value * 100:.1f}%' if value is not None else '-'


def correction_answers(bundle, corrections_file, output):
    """Each correction's answer from one model: (label shown as text, probability for it)."""
    result = score(bundle, [corrections_file], 'train', output)
    answers = {p['id']: ('yes' if p['p_yes'] >= .5 else 'no', p['p_yes']) for p in result.get('predictions', [])}
    for p in result.get('choice_predictions', []):
        answers[p['id']] = p['ranked'][0]
    return answers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', default=time.strftime('retrain-%Y%m%d-%H%M%S'), help='run folder under runs/')
    parser.add_argument('--no-corrections', action='store_true', help='train without data/corrections.jsonl (reproduces 0.4.1)')
    parser.add_argument('--check-only', action='store_true', help='check data/corrections.jsonl and stop')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--base', default=BASE, help='only for testing the script with a smaller model')
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--skip-baseline', action='store_true', help='do not download 0.4.1 to compare corrections')
    args = parser.parse_args()

    run = ROOT / 'runs' / args.name
    corrections = [] if args.no_corrections else read_corrections(CORRECTIONS)
    check_against_tests(corrections)
    if corrections:
        run.mkdir(parents=True, exist_ok=True)
        corrections_file = run / 'corrections.records.jsonl'
        corrections_file.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in corrections), encoding='utf-8')
    training_files = TRAINING_DATA + ([corrections_file] if corrections else [])
    counts = {}
    for row in pipeline.read_data(DEFAULT_DATA + training_files):  # also catches duplicates and conflicting labels
        counts[row['split']] = counts.get(row['split'], 0) + 1
    print(f'{len(corrections)} correction(s) from data/corrections.jsonl; records by split: {counts}', flush=True)
    if args.check_only:
        return

    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'cuda':
        memory = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f'GPU: {torch.cuda.get_device_name(0)}, {memory:.0f} GB', flush=True)
        if memory < 35 and args.base == BASE:
            print('Warning: the 0.4.1 recipe was run on a 40 GB A100; this GPU may run out of memory.', flush=True)
    elif args.base == BASE:
        print('Warning: no NVIDIA GPU found. Training the full model this way may take days or run out of memory.', flush=True)

    started = time.monotonic()
    extra = [str(f) for f in training_files]
    pipeline.train(SimpleNamespace(output=str(run / 'checkpoint'), extra_data=extra, data=None, epochs=args.epochs,
                                   batch_size=args.batch_size, learning_rate=1e-5, seed=42, device=device, start=None,
                                   resume=False, base=args.base, revision=args.revision, nli_head=True, template=2,
                                   gradient_checkpointing=False, fixed_shapes=False))
    model_id = f'decisiongator-0.4.1-{args.name}'
    pipeline.export(SimpleNamespace(checkpoint=str(run / 'checkpoint/best'), output=str(run / 'float'), model_id=model_id))
    bundle = run / 'model'
    subprocess.run([sys.executable, str(ROOT / 'training/quant/mixed_quant.py'), str(run / 'float'), str(bundle), '6', '17', '--reduce-range'], check=True)
    pipeline.evaluate(SimpleNamespace(command='calibrate', bundle=str(bundle), data=None, extra_data=extra,
                                      split='calibration', output=str(run / 'calibration.json')))

    # The native libraries check the ONNX Runtime they load against this manifest, so carry every platform's hash.
    manifest = json.loads((bundle / 'manifest.json').read_text())
    manifest['model_id'] = model_id
    for platform in ('linux-x64', 'macos-arm64', 'windows-x64'):
        released = json.loads((ROOT / f'released/{platform}/manifest.json').read_text())
        manifest['sha256'].update({k: v for k, v in released['sha256'].items() if 'onnxruntime' in k})
        manifest.setdefault('parity', released.get('parity'))
    manifest['corrections'] = len(corrections)
    (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    lines = [f'# Retrained model: {model_id}', '', f'{len(corrections)} correction(s); trained on {device}; '
             f'{(time.monotonic() - started) / 60:.0f} minutes.', '', '| Held-out test | 0.4.1 | New |', '| --- | ---: | ---: |']
    for label, files in TESTS.items():
        new = score(bundle, files, 'test', run / f'test-{PUBLISHED[label]}.json')
        published = json.loads((ROOT / f'reports/v4.1/{PUBLISHED[label]}.json').read_text())
        lines.append(f'| {label} | {percent(published)} | {percent(new)} |')
    if corrections:
        new_answers = correction_answers(bundle, corrections_file, run / 'corrections-new.json')
        old_answers = {} if args.skip_baseline else correction_answers(fetch_baseline(ROOT / 'runs/baseline-0.4.1'),
                                                                       corrections_file, run / 'corrections-old.json')
        lines += ['', '| Your correction | Wanted | 0.4.1 | New |', '| --- | --- | --- | --- |']
        for row in corrections:
            def shown(answers):
                if row['id'] not in answers:
                    return '-'
                value, p = answers[row['id']]
                return f'{row["options"][value] if row["label_type"] == "choice" else value} ({p:.2f})'
            wanted = row['options'][row['label']] if row['label_type'] == 'choice' else ('yes' if row['label'] else 'no')
            lines.append(f'| {row["content"][:50]} | {wanted} | {shown(old_answers)} | {shown(new_answers)} |')
        lines += ['', 'Corrections were trained on, so getting them right shows the model learned them, '
                  'not that it generalizes. The held-out tests above measure that.']
    report = '\n'.join(lines) + '\n'
    (run / 'results.md').write_text(report, encoding='utf-8')
    archive = run / 'decisiongator-model.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as z:
        for name in ('model.onnx', 'tokenizer.json', 'manifest.json'):
            z.write(bundle / name, name)
        z.writestr('results.md', report)
    shutil.rmtree(run / 'float', ignore_errors=True)
    print('\n' + report)
    print(f'Model: {archive}\nUnzip it into your DecisionGator bundle folder, replacing model.onnx, tokenizer.json and manifest.json.')


if __name__ == '__main__':
    main()
