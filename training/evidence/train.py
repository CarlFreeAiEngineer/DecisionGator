"""Three-way evidence-check model: supported / unsupported / not_claim. Design: specs/evidence-check.md.

Train on a GPU:   uv run python -m training.evidence.train --output runs/evidence-1 --export
Check the code:   uv run python -m training.evidence.train --smoke
Count truncation: uv run python -m training.evidence.train --truncation-only

metrics.json holds numbers only. Record text is never printed or written anywhere.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('HF_HOME', str(ROOT / 'tools/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
BASE = 'MoritzLaurer/ModernBERT-large-zeroshot-v2.0'
REVISION = 'a51e07b524299e309dd2b88d48b0cfa2bd9ec598'
LABELS = ['supported', 'unsupported', 'not_claim']
SPLITS = ('train', 'validation', 'calibration', 'test', 'test_unseen')
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_DATA = [ROOT / 'data/evidence/invented.jsonl',
                ROOT.parent / 'RealAiDating/evals/memory/evidence-labels/real-replies.jsonl.gz']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- data

def read_file(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    rows = []
    with opener(path, 'rt', encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # the invented file may be mid-append; skip a half-written last line
            if row.get('needs_review'):
                continue
            if row.get('label') not in LABELS or row.get('split') not in SPLITS:
                raise ValueError(f'{path.name}:{number}: bad label or split')
            for key in ('evidence', 'context', 'statement'):
                if not isinstance(row.get(key), str):
                    raise ValueError(f'{path.name}:{number}: missing {key}')
            row['source_file'] = Path(path).name
            rows.append(row)
    return rows


def load_data(paths):
    rows = []
    for path in paths:
        rows += read_file(Path(path))
    if not rows:
        raise ValueError('No data found')
    return rows


def first_segment(row):
    return f'Sentence: {row["statement"]}\nIn the reply: {row["context"]}'


def encode_rows(tokenizer, rows, max_tokens):
    """Add row['ids'] (input ids) and flags row['cut_evidence'], row['cut_first'].
    Only the evidence is cut, from the front, so its end (the most recent part) is kept."""
    special = len(tokenizer.build_inputs_with_special_tokens([], []))
    for row in rows:
        a = tokenizer(first_segment(row), add_special_tokens=False)['input_ids']
        b = tokenizer('Evidence:\n' + row['evidence'], add_special_tokens=False)['input_ids']
        row['cut_first'] = False
        if len(a) > max_tokens // 2:  # a pathological first segment must not eat the evidence budget
            a, row['cut_first'] = a[:max_tokens // 2], True
        room = max_tokens - special - len(a)
        row['cut_evidence'] = len(b) > room
        if row['cut_evidence']:
            b = b[len(b) - room:]
        row['ids'] = tokenizer.build_inputs_with_special_tokens(a, b)
    return rows


def make_batches(rows, tokens_per_batch, rng, shuffle=True):
    """Length-bucketed batches within a padded-token budget. Returns lists of rows."""
    order = sorted(rows, key=lambda r: len(r['ids']))
    if shuffle:  # jitter so equal-length rows do not always batch the same way
        order = sorted(rows, key=lambda r: len(r['ids']) + rng.random())
    batches, current, longest = [], [], 0
    for row in order:
        n = len(row['ids'])
        if current and max(longest, n) * (len(current) + 1) > tokens_per_batch:
            batches.append(current)
            current, longest = [], 0
        current.append(row)
        longest = max(longest, n)
    if current:
        batches.append(current)
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(batch, pad_id, device):
    import torch
    n = max(len(r['ids']) for r in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, r in enumerate(batch):
        ids[i, :len(r['ids'])] = torch.tensor(r['ids'])
        mask[i, :len(r['ids'])] = 1
    return ids.to(device), mask.to(device)


# ---------------------------------------------------------------- model

def build_model(args, tokenizer):
    import torch
    from transformers import AutoModelForSequenceClassification, ModernBertConfig, ModernBertForSequenceClassification
    if args.base == 'tiny':
        config = ModernBertConfig(vocab_size=len(tokenizer), hidden_size=64, num_hidden_layers=3, num_attention_heads=2,
                                  intermediate_size=128, num_labels=2, pad_token_id=tokenizer.pad_token_id,
                                  bos_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.sep_token_id,
                                  cls_token_id=tokenizer.cls_token_id, sep_token_id=tokenizer.sep_token_id)
        model = ModernBertForSequenceClassification(config)
        entail, not_entail = 0, 1
    else:
        options = {'revision': args.revision} if args.base == BASE else {}
        # Flash attention skips padding and is about 8x faster at 3,500 tokens on an L4 (measured 2026-11-15).
        # Install its prebuilt wheel into the venv and run with `uv run --no-sync`, or uv removes it again.
        import importlib.util
        if torch.cuda.is_available() and importlib.util.find_spec('flash_attn'):
            options['attn_implementation'] = 'flash_attention_2'
        print('attention:', options.get('attn_implementation', 'default'), flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.base, **options)
        names = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
        entail, not_entail = names.get('entailment', 0), names.get('not_entailment', 1)
    old = model.classifier
    head = torch.nn.Linear(old.in_features, 3)
    torch.nn.init.normal_(head.weight, std=0.02)
    torch.nn.init.zeros_(head.bias)
    with torch.no_grad():
        head.weight[0].copy_(old.weight[entail]); head.bias[0] = old.bias[entail]
        head.weight[1].copy_(old.weight[not_entail]); head.bias[1] = old.bias[not_entail]
    model.classifier = head
    model.num_labels = model.config.num_labels = 3
    model.config.id2label = dict(enumerate(LABELS))
    model.config.label2id = {n: i for i, n in enumerate(LABELS)}
    return model


def predict_logits(model, rows, device, pad_id, tokens_per_batch, autocast):
    import torch
    model.eval()
    out = [None] * len(rows)
    index = {id(r): i for i, r in enumerate(rows)}
    with torch.no_grad():
        for batch in make_batches(rows, tokens_per_batch * 2, random.Random(0), shuffle=False):
            ids, mask = collate(batch, pad_id, device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
                logits = model(input_ids=ids, attention_mask=mask).logits.float().cpu()
            for r, l in zip(batch, logits):
                out[index[id(r)]] = l
    return torch.stack(out) if out else torch.zeros((0, 3))


# ---------------------------------------------------------------- metrics

def class_metrics(labels, predicted):
    n = len(labels)
    result = {'count': n, 'accuracy': sum(a == b for a, b in zip(labels, predicted)) / n if n else None}
    f1s = []
    for c, name in enumerate(LABELS):
        tp = sum(1 for a, b in zip(labels, predicted) if a == c and b == c)
        fp = sum(1 for a, b in zip(labels, predicted) if a != c and b == c)
        fn = sum(1 for a, b in zip(labels, predicted) if a == c and b != c)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fn) else None
        result[name] = {'support': tp + fn, 'precision': precision, 'recall': recall, 'f1': f1}
        if f1 is not None:
            f1s.append(f1)
    result['macro_f1'] = sum(f1s) / len(f1s) if f1s else None
    return result


def redesign_metrics(rows, probs):
    """Real records only. Flag = p(unsupported) >= t. Messages are grouped by run + question."""
    real = [(r, p) for r, p in zip(rows, probs) if r.get('kind') == 'real']
    if not real:
        return None
    unsupported = [p[1] for r, p in real if r['label'] == 'unsupported']
    groups = {}
    for r, p in real:
        g = groups.setdefault((r.get('run'), r.get('question')), {'bad': False, 'flags': []})
        g['bad'] |= r['label'] == 'unsupported'
        g['flags'].append(p[1])
    clean = [g for g in groups.values() if not g['bad']]
    result = {'unsupported_sentences': len(unsupported), 'messages': len(groups), 'clean_messages': len(clean), 'by_threshold': {}}
    for t in THRESHOLDS:
        result['by_threshold'][str(t)] = {
            'unsupported_caught': sum(p >= t for p in unsupported) / len(unsupported) if unsupported else None,
            'clean_messages_flagged': sum(any(f >= t for f in g['flags']) for g in clean) / len(clean) if clean else None}
    return result


def evaluate(rows, logits, temperature=1.0):
    """Metrics per source file for one split. Returns {source: {...}}."""
    import torch
    probs = torch.softmax(logits / temperature, dim=-1).tolist()
    result = {}
    for source in sorted({r['source_file'] for r in rows}):
        sel = [i for i, r in enumerate(rows) if r['source_file'] == source]
        sub = [rows[i] for i in sel]
        sub_probs = [probs[i] for i in sel]
        predicted = [max(range(3), key=lambda c: p[c]) for p in sub_probs]
        labels = [LABELS.index(r['label']) for r in sub]
        m = class_metrics(labels, predicted)
        red = redesign_metrics(sub, sub_probs)
        if red:
            m['redesign'] = red
        kinds = {}
        for r, y, p in zip(sub, labels, predicted):
            if r.get('kind') != 'real' and y == 1:
                k = kinds.setdefault(r.get('kind') or 'none', [0, 0])
                k[0] += p == 1
                k[1] += 1
        if kinds:
            m['unsupported_recall_by_kind'] = {k: {'count': v[1], 'recall': v[0] / v[1]} for k, v in sorted(kinds.items())}
        result[source] = m
    return result


def fit_temperature(logits, labels):
    import torch
    if len(labels) == 0:
        return 1.0
    y = torch.tensor(labels)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20))


def truncation_stats(rows):
    by_source = {}
    for r in rows:
        s = by_source.setdefault(r['source_file'], {'count': 0, 'evidence_truncated': 0, 'first_segment_truncated': 0})
        s['count'] += 1
        s['evidence_truncated'] += r['cut_evidence']
        s['first_segment_truncated'] += r['cut_first']
    for s in by_source.values():
        s['evidence_truncated_share'] = s['evidence_truncated'] / s['count']
    return by_source


# ---------------------------------------------------------------- train

def parse_weights(items):
    weights = {}
    for item in items or []:
        name, _, w = item.rpartition('=')
        weights[Path(name).name] = float(w)
    return weights


def smoke_sample(rows, rng):
    picked = []
    for split, n in (('train', 20), ('validation', 8), ('calibration', 4), ('test', 4), ('test_unseen', 2)):
        pool = [r for r in rows if r['split'] == split]
        rng.shuffle(pool)
        picked += pool[:n]
    return picked


def train(args):
    import torch
    from transformers import AutoTokenizer
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    paths = [Path(p) for p in args.data] if args.data else [p for p in DEFAULT_DATA if p.exists()]
    if args.smoke:
        args.max_tokens, args.epochs = 256, 1
        args.base = args.base or 'tiny'
        args.output = args.output or tempfile.mkdtemp(prefix='evidence-smoke-')
        args.export = True
    args.base = args.base or BASE
    rows = load_data(paths)
    if args.smoke:
        rows = smoke_sample(rows, rng)
    tok_source, tok_options = (BASE, {'revision': REVISION}) if args.base == 'tiny' else (args.base, {'revision': args.revision} if args.base == BASE else {})
    tokenizer = AutoTokenizer.from_pretrained(tok_source, **tok_options)
    encode_rows(tokenizer, rows, args.max_tokens)
    stats = truncation_stats(rows)
    print('truncation (counts):', json.dumps(stats), flush=True)
    if args.truncation_only:
        return
    by_split = {s: [r for r in rows if r['split'] == s] for s in SPLITS}
    if not by_split['train'] or not by_split['validation']:
        raise ValueError('Both train and validation data required')
    print('records per split:', {s: len(v) for s, v in by_split.items()}, flush=True)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output exists and is not empty: choose a new directory')
    output.mkdir(parents=True, exist_ok=True)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    autocast = device == 'cuda'
    model = build_model(args, tokenizer).to(device)
    if args.gradient_checkpointing and not args.smoke:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    pad_id = tokenizer.pad_token_id
    weights = parse_weights(args.weight)

    def epoch_rows():
        rows_out = []
        for r in by_split['train']:
            w = weights.get(r['source_file'], 1.0)
            copies = int(w) + (rng.random() < w - int(w))
            rows_out += [r] * copies
        return rows_out

    def plan():  # list of optimizer steps, each a list of micro-batches covering about --batch-examples examples
        steps, current, count = [], [], 0
        for batch in make_batches(epoch_rows(), args.tokens_per_batch, rng):
            current.append(batch); count += len(batch)
            if count >= args.batch_examples:
                steps.append(current); current, count = [], 0
        if current:
            steps.append(current)
        return steps

    first_plan = plan()
    total_steps = max(1, len(first_plan) * args.epochs)
    if args.smoke:
        total_steps = 1
    warmup = max(1, int(round(0.06 * total_steps)))
    decay = ['bias', 'norm']
    named = list(model.named_parameters())
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in named if not any(d in n for d in decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in named if any(d in n for d in decay)], 'weight_decay': 0.0}], lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    config = {'base': args.base, 'revision': args.revision if args.base == BASE else None, 'seed': args.seed,
              'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay, 'epochs': args.epochs,
              'max_tokens': args.max_tokens, 'tokens_per_batch': args.tokens_per_batch, 'batch_examples': args.batch_examples,
              'weights': weights, 'device': device, 'torch': torch.__version__, 'smoke': args.smoke,
              'data': {p.name: digest(p) for p in paths}, 'train_script_sha256': digest(__file__),
              'records_per_split': {s: len(v) for s, v in by_split.items()}, 'truncation': stats}
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

    best_f1, best_epoch, history, step = -1.0, 0, [], 0
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        steps = first_plan if epoch == 0 else plan()
        loss_sum, seen = 0.0, 0
        for micro in steps:
            if step >= total_steps:
                break
            n = sum(len(b) for b in micro)
            optimizer.zero_grad(set_to_none=True)
            for batch in micro:
                ids, mask = collate(batch, pad_id, device)
                y = torch.tensor([LABELS.index(r['label']) for r in batch], device=device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
                    logits = model(input_ids=ids, attention_mask=mask).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), y, reduction='sum')
                (loss / n).backward()
                loss_sum += loss.item(); seen += len(batch)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); step += 1
            if step % 20 == 0:
                print(f'epoch {epoch + 1} step {step}/{total_steps} loss {loss_sum / max(1, seen):.4f}', flush=True)
        val_logits = predict_logits(model, by_split['validation'], device, pad_id, args.tokens_per_batch, autocast)
        val = evaluate(by_split['validation'], val_logits)
        overall = class_metrics([LABELS.index(r['label']) for r in by_split['validation']], val_logits.argmax(-1).tolist())
        record = {'epoch': epoch + 1, 'train_loss': loss_sum / max(1, seen), 'validation': overall,
                  'validation_by_source': val, 'elapsed_seconds': time.monotonic() - started}
        history.append(record)
        print(json.dumps({'epoch': epoch + 1, 'train_loss': record['train_loss'], 'val_acc': overall['accuracy'],
                          'val_macro_f1': overall['macro_f1']}), flush=True)
        model.save_pretrained(output / f'epoch-{epoch + 1}')
        tokenizer.save_pretrained(output / f'epoch-{epoch + 1}')
        if (overall['macro_f1'] or 0) > best_f1:
            best_f1, best_epoch = overall['macro_f1'] or 0, epoch + 1
    best_dir = output / f'epoch-{best_epoch}'
    print(f'best epoch {best_epoch} (validation macro-F1 {best_f1:.4f})', flush=True)

    # Final evaluation with the best checkpoint.
    del model, optimizer
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    logits = {s: predict_logits(model, by_split[s], device, pad_id, args.tokens_per_batch, autocast) for s in SPLITS if by_split[s]}
    temperature = 1.0
    if by_split['calibration']:
        temperature = fit_temperature(logits['calibration'], [LABELS.index(r['label']) for r in by_split['calibration']])
    print(f'temperature {temperature:.4f}', flush=True)
    final = {s: {'uncalibrated': evaluate(by_split[s], logits[s]), 'calibrated': evaluate(by_split[s], logits[s], temperature)}
             for s in SPLITS if s != 'train' and by_split[s]}
    metrics = {'best_epoch': best_epoch, 'temperature': temperature, 'history': history, 'final': final,
               'truncation': stats, 'config': config}
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    del model
    if args.export:
        export(args, best_dir, output, tokenizer, temperature, by_split, config)
    print(f'done; metrics in {output / "metrics.json"}', flush=True)


# ---------------------------------------------------------------- export

def export(args, checkpoint, output, tokenizer, temperature, by_split, config):
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForSequenceClassification
    target = output / 'onnx'
    target.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, attn_implementation='eager', torch_dtype=torch.float32).eval().cpu()
    tokenizer.backend_tokenizer.no_padding()
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.backend_tokenizer.save(str(target / 'tokenizer.json'))
    pool = [r for s in ('validation', 'test', 'train') for r in by_split[s]]
    samples = pool[:5]
    pad_id = tokenizer.pad_token_id
    ids, mask = collate(samples[:2], pad_id, 'cpu')

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = model

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    with torch.no_grad():
        torch.onnx.export(Wrapper().eval(), (ids, mask), str(target / 'model.onnx'), input_names=['input_ids', 'attention_mask'],
                          output_names=['logits'], opset_version=17, dynamo=False,
                          dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}, 'attention_mask': {0: 'batch', 1: 'sequence'},
                                        'logits': {0: 'batch'}})
    session = ort.InferenceSession(str(target / 'model.onnx'), providers=['CPUExecutionProvider'])
    worst = 0.0
    for r in samples:  # one at a time: also checks different lengths through the dynamic axes
        i, m = collate([r], pad_id, 'cpu')
        actual = session.run(None, {'input_ids': i.numpy(), 'attention_mask': m.numpy()})[0]
        with torch.no_grad():
            expected = model(input_ids=i, attention_mask=m).logits.numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
        worst = max(worst, float(np.max(np.abs(actual - expected))))
    manifest = {'format_version': 1, 'model_id': args.model_id, 'labels': LABELS, 'max_tokens': args.max_tokens,
                'temperature': temperature, 'base_model': config['base'], 'base_revision': config['revision'],
                'input_template': {'first': 'Sentence: {statement}\nIn the reply: {context}', 'second': 'Evidence:\n{evidence}',
                                   'truncation': 'evidence only, cut from the front'},
                'sha256': {n: digest(target / n) for n in ('model.onnx', 'tokenizer.json')},
                'checkpoint_sha256': digest(Path(checkpoint) / 'model.safetensors'),
                'train_script_sha256': config['train_script_sha256'], 'data_sha256': config['data'],
                'smoke': config['smoke']}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {target}; max logit difference over {len(samples)} examples: {worst:.6g}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', action='append', help='jsonl or jsonl.gz file; repeatable')
    parser.add_argument('--weight', action='append', metavar='FILE=W', help='upweight a source file (repeat W times per epoch)')
    parser.add_argument('--output', help='run directory (must be new)')
    parser.add_argument('--base', help=f'foundation model (default {BASE}; "tiny" is a random toy model for tests)')
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--max-tokens', type=int, default=2560)
    parser.add_argument('--tokens-per-batch', type=int, default=14000)
    parser.add_argument('--batch-examples', type=int, default=16, help='examples per optimizer step (gradient accumulation)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--model-id', default='evidence-check-0.1')
    parser.add_argument('--smoke', action='store_true', help='tiny CPU run (40 records, 256 tokens, one step) incl. export')
    parser.add_argument('--truncation-only', action='store_true', help='only count truncated examples at --max-tokens')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
