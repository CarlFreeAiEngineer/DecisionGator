"""Evidence check, one model pass per MESSAGE: every sentence of a reply is judged in a single call.
Same labels, data, metrics and recipe as train.py (see specs/evidence-check.md); only the input shape differs.

Input:  "Reply: [1] sentence one [2] sentence two ..." then "Evidence:\\n<evidence>" (evidence cut from the front).
Output: three-way logits at each marker token (ModernBERT's own head layers, then a 3-way linear layer).

Train on a GPU:   uv run python -m training.evidence.train_message --output runs/evidence-m1 --export
Check the code:   uv run python -m training.evidence.train_message --smoke
Count messages:   uv run python -m training.evidence.train_message --stats

metrics.json and predictions.jsonl hold numbers and ids only. Record text is never printed or written anywhere.
"""
import argparse
import json
from pathlib import Path
import random
import tempfile
import time

from training.evidence import train as T
from training.evidence.train import LABELS, SPLITS, class_metrics, digest, evaluate, fit_temperature

IGNORE = -100


# ---------------------------------------------------------------- messages

NEW_KINDS = {'added_detail', 'never_told', 'removed', 'added_clause'}


def is_flip(r):
    return bool(r.get('flip_of') or r.get('flipped_from') or r.get('variant_of') or r.get('kind') == 'removed')


def mark_scope(rows):
    """Only clearly new information counts as unsupported. Borderline sentences, and unsupported ones that retell a
    real event wrongly (wrong day, order, number, person), get r['ignore'] = True: they stay in the reply as context
    but carry no label, in training and in scoring. Returns the number ignored."""
    n = 0
    for r in rows:
        told_wrong = r['label'] == 'unsupported' and (r.get('type') != 'new' if r.get('kind') == 'real' else r.get('kind') not in NEW_KINDS)
        r['ignore'] = bool(r.get('borderline') or told_wrong or r.get('unlabeled'))
        n += r['ignore']
    return n


def sort_key(item):
    position, r = item
    return (r.get('sentence_index') if r.get('sentence_index') is not None else position, position)


def build_messages(rows, keep_unlabeled=False):
    """Group sentence records into messages. Returns a list of dicts:
    {'id', 'split', 'source_file', 'evidence', 'sentences': [(text, record or None)]}  (None = context only, no label).
    A reply is (group_id, run, question, context). Ordinary records of a reply with the same evidence form
    one message. A flipped record (edited evidence) forms a message with the other sentences of the reply as context.
    A variant record (variant_of = id of an original sentence, same evidence) forms its own message: the reply with
    that one sentence replaced. Records marked ignore (see mark_scope) are context only."""
    def reply_key(r):
        return (r['group_id'], r.get('run'), r.get('question'), r['context'])

    base = {}
    for position, r in enumerate(rows):
        if not is_flip(r):
            base.setdefault(reply_key(r), []).append((position, r))
    for key in base:
        base[key] = [r for _, r in sorted(base[key], key=sort_key)]
    messages = {}
    for r in rows:
        if is_flip(r):
            continue
        m = messages.setdefault((reply_key(r), r['evidence'], None), {'evidence': r['evidence'], 'sentences': [], 'any': r})
        m['sentences'].append((r['statement'], None if r.get('ignore') else r))
    for r in rows:
        if not is_flip(r) or r.get('ignore'):
            continue
        key = reply_key(r)
        mk = (key, r['evidence'], r['id'] if r.get('variant_of') else None)
        m = None if r.get('variant_of') else messages.get(mk)
        if m is None:
            m = messages[mk] = {'evidence': r['evidence'], 'sentences': [(b['statement'], None) for b in base.get(key, [])],
                                                  'origins': [b['id'] for b in base.get(key, [])]}
        origin = r.get('flip_of') or r.get('flipped_from') or r.get('variant_of')
        origins = m.get('origins', [])
        if origin in origins:
            m['sentences'][origins.index(origin)] = (r['statement'], r)
        else:
            m['sentences'].append((r['statement'], r))
            origins.append(None)
    out = []
    for m in messages.values():
        labeled = [r for _, r in m['sentences'] if r is not None]
        first = labeled[0] if labeled else m.get('any') if keep_unlabeled else None
        if first is None or (not labeled and first['split'] != 'train'):  # unlabeled messages only teach (see --teacher)
            continue
        m.pop('origins', None); m.pop('any', None)
        m['id'], m['split'], m['source_file'] = first['id'], first['split'], first['source_file']
        out.append(m)
    return out


def message_stats(messages):
    by_source = {}
    for m in messages:
        s = by_source.setdefault(m['source_file'], {'messages': 0, 'sentences': 0, 'labeled_sentences': 0})
        s['messages'] += 1
        s['sentences'] += len(m['sentences'])
        s['labeled_sentences'] += sum(r is not None for _, r in m['sentences'])
    for s in by_source.values():
        s['mean_sentences_per_message'] = round(s['sentences'] / s['messages'], 3)
        s['mean_labeled_per_message'] = round(s['labeled_sentences'] / s['messages'], 3)
    return by_source


def reply_text(sentences):
    """'Reply: [1] one [2] two' and the character position of each marker's opening bracket."""
    text, positions = 'Reply:', []
    for i, s in enumerate(sentences, 1):
        text += ' '
        positions.append(len(text))
        text += f'[{i}] {s}'
    return text, positions


def encode_messages(tokenizer, messages, max_tokens):
    """Add m['ids'], m['positions'] (token index of each marker's first token), m['targets'] (label id or IGNORE),
    m['rows'] (labeled records in marker order) and flags. Only the evidence is cut, from the front. If the reply
    itself is longer than half the budget, its last sentences are dropped (counted in m['dropped'])."""
    special = len(tokenizer.build_inputs_with_special_tokens([], []))
    for m in messages:
        sentences = list(m['sentences'])
        m['dropped'] = 0
        while True:
            text, starts = reply_text([s for s, _ in sentences])
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            if len(enc['input_ids']) <= max_tokens // 2 or len(sentences) <= 1:
                break
            m['dropped'] += sentences.pop()[1] is not None
        a = enc['input_ids']
        m['cut_first'] = len(a) > max_tokens // 2
        if m['cut_first']:
            a = a[:max_tokens // 2]
        offsets = enc['offset_mapping']
        positions = []
        for c in starts:
            positions.append(next((t for t, (s, e) in enumerate(offsets) if e > c), len(a) - 1))
        positions = [min(p, len(a) - 1) for p in positions]
        b = tokenizer('Evidence:\n' + m['evidence'], add_special_tokens=False)['input_ids']
        room = max_tokens - special - len(a)
        m['cut_evidence'] = len(b) > room
        if m['cut_evidence']:
            b = b[len(b) - room:]
        m['ids'] = tokenizer.build_inputs_with_special_tokens(a, b)
        shift = m['ids'].index(a[0]) if a else 0
        assert m['ids'][shift:shift + len(a)] == a
        m['positions'] = [p + shift for p in positions]
        m['targets'] = [LABELS.index(r['label']) if r is not None else IGNORE for _, r in sentences]
        m['rows'] = [r for _, r in sentences if r is not None]
    return messages


def collate_messages(batch, pad_id, device):
    import torch
    ids, mask = T.collate(batch, pad_id, device)
    k = max(len(m['positions']) for m in batch)
    pos = torch.zeros((len(batch), k), dtype=torch.long)
    targets = torch.full((len(batch), k), IGNORE, dtype=torch.long)
    for i, m in enumerate(batch):
        pos[i, :len(m['positions'])] = torch.tensor(m['positions'])
        targets[i, :len(m['targets'])] = torch.tensor(m['targets'])
    return ids, mask, pos.to(device), targets.to(device)


def teacher_probabilities(teacher_dir, messages, device, pad_id, tokens_per_batch, autocast, temperature):
    """Set m['soft'] = [markers, 3] probabilities from a trained model, for every sentence of every message."""
    import torch
    from transformers import AutoModelForSequenceClassification
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_dir).to(device).eval()
    with torch.no_grad():
        for batch in T.make_batches(messages, tokens_per_batch, random.Random(0), shuffle=False):
            ids, mask, pos, _ = collate_messages(batch, pad_id, device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
                probs = torch.softmax(message_logits(teacher, ids, mask, pos).float() / temperature, dim=-1).cpu()
            for m, p in zip(batch, probs):
                m['soft'] = p[:len(m['positions'])]
    del teacher
    if device == 'cuda':
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- model

def message_logits(model, ids, mask, pos):
    """Logits [batch, markers, 3]: the final hidden state at each marker, through the base model's own head
    (dense, activation, norm), then the 3-way linear layer."""
    hidden = model.model(input_ids=ids, attention_mask=mask)[0]
    picked = hidden.gather(1, pos.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))
    return model.classifier(model.drop(model.head(picked)))


def predict_message_logits(model, messages, device, pad_id, tokens_per_batch, autocast):
    """Per message, a [labeled markers, 3] tensor (context-only markers left out)."""
    import torch
    model.eval()
    out = [None] * len(messages)
    index = {id(m): i for i, m in enumerate(messages)}
    with torch.no_grad():
        for batch in T.make_batches(messages, tokens_per_batch * 2, random.Random(0), shuffle=False):
            ids, mask, pos, targets = collate_messages(batch, pad_id, device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
                logits = message_logits(model, ids, mask, pos).float().cpu()
            for m, l in zip(batch, logits):
                keep = [i for i, t in enumerate(m['targets']) if t != IGNORE]
                out[index[id(m)]] = l[keep]
    return out


def flatten(messages, per_message):
    import torch
    rows = [r for m in messages for r in m['rows']]
    logits = torch.cat(per_message) if per_message else torch.zeros((0, 3))
    return rows, logits


# ---------------------------------------------------------------- train

def train(args):
    import torch
    from transformers import AutoTokenizer
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    paths = [Path(p) for p in args.data] if args.data else [p for p in T.DEFAULT_DATA if p.exists()]
    if args.smoke:
        args.max_tokens, args.epochs = 256, 1
        args.base = args.base or 'tiny'
        args.output = args.output or tempfile.mkdtemp(prefix='evidence-message-smoke-')
        args.export = True
    args.base = args.base or T.BASE
    rows = T.load_data(paths)
    if args.relabel:
        T.apply_relabels(rows, args.relabel)
    if args.only_new_information:
        print('ignored (borderline or told wrong):', mark_scope(rows), 'of', len(rows), flush=True)
    messages = build_messages(rows, keep_unlabeled=bool(args.teacher))
    stats_by_source = message_stats(messages)
    print('messages (counts):', json.dumps(stats_by_source), flush=True)
    if args.stats:
        return
    if args.smoke:
        picked = []
        for split, n in (('train', 12), ('validation', 6), ('calibration', 3), ('test', 3), ('test_unseen', 2)):
            pool = [m for m in messages if m['split'] == split]
            rng.shuffle(pool)
            picked += pool[:n]
        messages = picked
    tok_source, tok_options = (T.BASE, {'revision': T.REVISION}) if args.base == 'tiny' else (args.base, {'revision': args.revision} if args.base == T.BASE else {})
    tokenizer = AutoTokenizer.from_pretrained(tok_source, **tok_options)
    encode_messages(tokenizer, messages, args.max_tokens)
    trunc = {}
    for m in messages:
        s = trunc.setdefault(m['source_file'], {'messages': 0, 'evidence_truncated': 0, 'first_segment_truncated': 0, 'sentences_dropped': 0})
        s['messages'] += 1
        s['evidence_truncated'] += m['cut_evidence']
        s['first_segment_truncated'] += m['cut_first']
        s['sentences_dropped'] += m['dropped']
    print('truncation (counts):', json.dumps(trunc), flush=True)
    by_split = {s: [m for m in messages if m['split'] == s] for s in SPLITS}
    if not by_split['train'] or not by_split['validation']:
        raise ValueError('Both train and validation data required')
    sentences_per_split = {s: sum(len(m['rows']) for m in v) for s, v in by_split.items()}
    print('messages per split:', {s: len(v) for s, v in by_split.items()}, 'labeled sentences:', sentences_per_split, flush=True)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output exists and is not empty: choose a new directory')
    output.mkdir(parents=True, exist_ok=True)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    autocast = device == 'cuda'
    model = T.build_model(args, tokenizer).to(device)
    if args.gradient_checkpointing and not args.smoke:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    pad_id = tokenizer.pad_token_id
    weights = T.parse_weights(args.weight)
    class_weights = T.parse_class_weights(args.class_weight)
    class_tensor = torch.tensor([class_weights.get(l, 1.0) for l in LABELS], device=device) if class_weights else None

    if args.teacher:
        teacher_probabilities(args.teacher, by_split['train'], device, pad_id, args.tokens_per_batch, autocast, args.teacher_temperature)
        print('teacher probabilities for', len(by_split['train']), 'training messages,',
              sum(len(m['positions']) for m in by_split['train']), 'sentences', flush=True)

    def epoch_messages():
        out = []
        for m in by_split['train']:
            w = weights.get(m['source_file'], 1.0)
            out += [m] * (int(w) + (rng.random() < w - int(w)))
        return out

    def plan():
        steps, current, count = [], [], 0
        for batch in T.make_batches(epoch_messages(), args.tokens_per_batch, rng):
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

    config = {'teacher': args.teacher, 'teacher_weight': args.teacher_weight, 'base': args.base, 'revision': args.revision if args.base == T.BASE else None, 'seed': args.seed,
              'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay, 'epochs': args.epochs,
              'max_tokens': args.max_tokens, 'tokens_per_batch': args.tokens_per_batch, 'batch_examples': args.batch_examples,
              'weights': weights, 'class_weights': class_weights, 'device': device, 'torch': torch.__version__, 'smoke': args.smoke,
              'data': {p.name: digest(p) for p in paths}, 'train_script_sha256': digest(__file__),
              'messages_per_split': {s: len(v) for s, v in by_split.items()}, 'labeled_sentences_per_split': sentences_per_split,
              'message_stats': stats_by_source, 'truncation': trunc}
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

    def validate_flat(model):
        rows_v, logits_v = flatten(by_split['validation'], predict_message_logits(model, by_split['validation'], device, pad_id, args.tokens_per_batch, autocast))
        return rows_v, logits_v

    best_f1, best_epoch, history, step = -1.0, 0, [], 0
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        steps = first_plan if epoch == 0 else plan()
        loss_sum, seen = 0.0, 0
        for micro in steps:
            if step >= total_steps:
                break
            n = sum(len(m['positions']) if args.teacher else sum(t != IGNORE for t in m['targets']) for b in micro for m in b)
            optimizer.zero_grad(set_to_none=True)
            for batch in micro:
                ids, mask, pos, targets = collate_messages(batch, pad_id, device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
                    logits = message_logits(model, ids, mask, pos)
                loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, 3), targets.reshape(-1), weight=class_tensor,
                                                         ignore_index=IGNORE, reduction='sum')
                if args.teacher:  # also match the teacher at every sentence, labeled or not
                    k = logits.size(1)
                    soft = torch.zeros((len(batch), k, 3))
                    have = torch.zeros((len(batch), k), dtype=torch.bool)
                    for i, m in enumerate(batch):
                        soft[i, :len(m['soft'])] = m['soft']; have[i, :len(m['soft'])] = True
                    log_p = torch.log_softmax(logits.float() / args.teacher_temperature, dim=-1)
                    kl = torch.nn.functional.kl_div(log_p, soft.to(device), reduction='none').sum(-1)
                    loss = loss + args.teacher_weight * (kl * have.to(device)).sum()
                (loss / max(1, n)).backward()
                loss_sum += loss.item(); seen += int((targets != IGNORE).sum())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); step += 1
            if step % 20 == 0:
                print(f'epoch {epoch + 1} step {step}/{total_steps} loss {loss_sum / max(1, seen):.4f}', flush=True)
        rows_v, logits_v = validate_flat(model)
        val = evaluate(rows_v, logits_v)
        overall = class_metrics([LABELS.index(r['label']) for r in rows_v], logits_v.argmax(-1).tolist())
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

    del model, optimizer
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    flat = {}
    for s in SPLITS:
        if by_split[s]:
            flat[s] = flatten(by_split[s], predict_message_logits(model, by_split[s], device, pad_id, args.tokens_per_batch, autocast))
    temperature = 1.0
    if by_split['calibration']:
        rows_c, logits_c = flat['calibration']
        temperature = fit_temperature(logits_c, [LABELS.index(r['label']) for r in rows_c])
    print(f'temperature {temperature:.4f}', flush=True)
    final = {s: {'uncalibrated': evaluate(flat[s][0], flat[s][1]), 'calibrated': evaluate(flat[s][0], flat[s][1], temperature)}
             for s in SPLITS if s != 'train' and by_split[s]}
    metrics = {'best_epoch': best_epoch, 'temperature': temperature, 'history': history, 'final': final,
               'truncation': trunc, 'config': config}
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    with open(output / 'predictions.jsonl', 'w', encoding='utf-8') as f:
        for s in final:
            rows_s, logits_s = flat[s]
            for r, p in zip(rows_s, torch.softmax(logits_s / temperature, dim=-1).tolist()):
                f.write(json.dumps({'id': r['id'], 'split': s, 'source': r['source_file'], 'label': r['label'], 'kind': r.get('kind'),
                                    'borderline': bool(r.get('borderline')), 'run': r.get('run'), 'question': r.get('question'),
                                    'probs': [round(x, 4) for x in p]}) + '\n')
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
    pool = [m for s in ('validation', 'test', 'train') for m in by_split[s]]
    samples = pool[:5]
    pad_id = tokenizer.pad_token_id
    # Two messages of different lengths and marker counts, so padding in both directions is traced.
    pair = sorted(pool[:8], key=lambda m: (len(m['positions']), len(m['ids'])))
    example = [pair[0], pair[-1]]
    ids, mask, pos, _ = collate_messages(example, pad_id, 'cpu')

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = model

        def forward(self, input_ids, attention_mask, marker_positions):
            return message_logits(self.inner, input_ids, attention_mask, marker_positions)

    with torch.no_grad():
        torch.onnx.export(Wrapper().eval(), (ids, mask, pos), str(target / 'model.onnx'),
                          input_names=['input_ids', 'attention_mask', 'marker_positions'], output_names=['logits'],
                          opset_version=17, dynamo=False,
                          dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}, 'attention_mask': {0: 'batch', 1: 'sequence'},
                                        'marker_positions': {0: 'batch', 1: 'markers'}, 'logits': {0: 'batch', 1: 'markers'}})
    session = ort.InferenceSession(str(target / 'model.onnx'), providers=['CPUExecutionProvider'])
    worst = 0.0
    for m in samples:  # one at a time: also checks different lengths and marker counts through the dynamic axes
        i, a, p, _ = collate_messages([m], pad_id, 'cpu')
        actual = session.run(None, {'input_ids': i.numpy(), 'attention_mask': a.numpy(), 'marker_positions': p.numpy()})[0]
        with torch.no_grad():
            expected = message_logits(model, i, a, p).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
        worst = max(worst, float(np.max(np.abs(actual - expected))))
    manifest = {'format_version': 1, 'model_id': args.model_id, 'labels': LABELS, 'max_tokens': args.max_tokens,
                'temperature': temperature, 'base_model': config['base'], 'base_revision': config['revision'],
                'input_format': {
                    'first': 'Reply: [1] sentence one [2] sentence two ... (one marker before each sentence, numbered from 1)',
                    'second': 'Evidence:\n{evidence}',
                    'truncation': 'evidence only, cut from the front; encode as [CLS] first [SEP] second [SEP]',
                    'marker_positions': 'int64 [batch, n_markers]: index in input_ids of the first token of each "[k]" marker; pad with 0',
                    'inputs': ['input_ids', 'attention_mask', 'marker_positions'], 'output': 'logits [batch, n_markers, 3]',
                    'note': 'one pass judges every sentence of the reply; rows for padded markers are meaningless'},
                'sha256': {n: digest(target / n) for n in ('model.onnx', 'tokenizer.json')},
                'checkpoint_sha256': digest(Path(checkpoint) / 'model.safetensors'),
                'train_script_sha256': config['train_script_sha256'], 'data_sha256': config['data'],
                'smoke': config['smoke']}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {target}; max logit difference over {len(samples)} messages: {worst:.6g}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', action='append', help='jsonl or jsonl.gz file; repeatable')
    parser.add_argument('--weight', action='append', metavar='FILE=W', help='upweight a source file (repeat W times per epoch)')
    parser.add_argument('--class-weight', action='append', metavar='LABEL=W', help='loss weight for a class; repeatable')
    parser.add_argument('--output', help='run directory (must be new)')
    parser.add_argument('--base', help=f'foundation model (default {T.BASE}; "tiny" is a random toy model for tests)')
    parser.add_argument('--revision', default=T.REVISION)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--relabel', action='append', default=[], help='reviewed labels by id (jsonl or jsonl.gz); repeatable')
    parser.add_argument('--tokens-per-batch', type=int, default=14000)
    parser.add_argument('--batch-examples', type=int, default=8, help='messages per optimizer step (gradient accumulation)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--model-id', default='evidence-check-message-0.1')
    parser.add_argument('--smoke', action='store_true', help='tiny CPU run (a few messages, 256 tokens, one step) incl. export')
    parser.add_argument('--only-new-information', action='store_true',
                        help='unsupported means clearly new information only; borderline and told-wrong sentences carry no label')
    parser.add_argument('--teacher', help='trained model directory (for example runs/x/epoch-3): the new model also learns '
                        'to match its probabilities on every training sentence, including records marked "unlabeled"')
    parser.add_argument('--teacher-weight', type=float, default=2.0)
    parser.add_argument('--teacher-temperature', type=float, default=1.0)
    parser.add_argument('--stats', action='store_true', help='only count messages and sentences per message')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
