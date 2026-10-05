"""Small, explicit training pipeline. Run with uv run decisiongator-train --help."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'tools/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
BASE = 'microsoft/MiniLM-L12-H384-uncased'
REVISION = '44acabbec0ef496f6dbc93adadea57f376b7c0ec'
MAX_TOKENS = 256
MAX_OPTIONS = 256
FIXED_SHAPES = False  # pad every batch to MAX_TOKENS; constant shapes keep the MPS allocator from fragmenting


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt(record):
    question = record['question']
    criteria = record.get('criteria')
    return question if criteria is None else question + '\nYes: ' + criteria['yes'] + '\nNo: ' + criteria['no']


def expand(rows):
    """Turn each choice row into one binary row per option (id '<id>#<k>', choice_of the parent). Binary rows pass through."""
    result = []
    for row in rows:
        if row['label_type'] == 'binary':
            result.append(row)
            continue
        for k, option in enumerate(row['options']):
            child = {**row, 'id': f'{row["id"]}#{k}', 'question': row['question'] + '\nAnswer: ' + option,
                      'label': int(k == row['label']), 'label_type': 'binary', 'choice_of': row['id']}
            child.pop('options', None)
            result.append(child)
    return result


def choice_metrics(children, probabilities):
    """Per-parent choice accuracy/log-loss from expanded binary rows and their predicted p_yes.
    p_yes = sigmoid(z) where z = (logit_yes-logit_no)/temperature, so log(p/(1-p)) recovers z exactly;
    softmax over a group's z picks the same argmax as the runtime's choice/rank regardless of temperature."""
    import numpy as np
    groups = {}
    for row, p in zip(children, probabilities):
        groups.setdefault(row['choice_of'], []).append((row, p))
    if not groups:
        return {}
    correct, families, log_losses, predictions = 0, {}, [], []
    for parent_id, items in groups.items():
        p = np.clip(np.array([x[1] for x in items]), 1e-7, 1 - 1e-7)
        z = np.log(p / (1 - p))
        weights = np.exp(z - z.max())
        probs = weights / weights.sum()
        choice_idx = int(np.argmax(probs))
        label_idx = next(i for i, (r, _) in enumerate(items) if r['label'] == 1)
        family = items[0][0]['task_family']
        counts = families.setdefault(family, [0, 0])
        counts[1] += 1
        if choice_idx == label_idx:
            correct += 1
            counts[0] += 1
        log_losses.append(float(-np.log(probs[label_idx])))
        predictions.append({'id': parent_id, 'label': label_idx,
                             'ranked': sorted(([i, float(pr)] for i, pr in enumerate(probs)), key=lambda t: (-t[1], t[0]))})
    return {'choice_accuracy': correct / len(groups), 'choice_count': len(groups),
            'choice_log_loss': float(np.mean(log_losses)),
            'choice_by_family': {f: {'count': c[1], 'accuracy': c[0] / c[1]} for f, c in families.items()},
            'choice_predictions': predictions}


def read_data(paths):
    rows, ids, groups, examples = [], set(), {}, set()
    for path in paths:
        for line_number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            location = f'{path}:{line_number}'
            if row.get('schema_version') != 1 or row.get('label_type') not in ('binary', 'choice'):
                raise ValueError(f'{location}: expected schema_version 1 and binary or choice label_type')
            for key in ('id', 'group_id', 'content', 'question', 'task_family', 'rationale', 'provenance', 'license', 'review_status'):
                if not row.get(key):
                    raise ValueError(f'{location}: missing {key}')
            if row['id'] in ids:
                raise ValueError(f'{location}: duplicate id {row["id"]}')
            ids.add(row['id'])
            if row['label_type'] == 'binary':
                if type(row.get('label')) is not int or row['label'] not in (0, 1):
                    raise ValueError(f'{location}: label must be integer 0 or 1')
            else:
                options = row.get('options')
                if not isinstance(options, list) or not (2 <= len(options) <= MAX_OPTIONS) or any(not isinstance(o, str) or not o.strip() for o in options):
                    raise ValueError(f'{location}: options must be 2 to {MAX_OPTIONS} nonempty strings')
                if type(row.get('label')) is not int or not (0 <= row['label'] < len(options)):
                    raise ValueError(f'{location}: label must be a valid option index')
            if row.get('split') not in ('train', 'validation', 'calibration', 'test'):
                raise ValueError(f'{location}: invalid split')
            for key in ('content', 'question'):
                if not isinstance(row[key], str) or not row[key].strip() or len(row[key].encode('utf-8'))>1_048_576:
                    raise ValueError(f'{location}: {key} must be nonempty text')
            criteria = row.get('criteria')
            if criteria is not None and (not isinstance(criteria, dict) or set(criteria) != {'yes', 'no'} or any(not isinstance(v, str) or not v.strip() for v in criteria.values())):
                raise ValueError(f'{location}: criteria must contain nonempty yes and no text')
            group = row['group_id']
            if group in groups and groups[group] != row['split']:
                raise ValueError(f'{location}: source group crosses splits')
            groups[group] = row['split']
            keys = [(row['content'].strip().casefold(), prompt(row).strip().casefold())] if row['label_type'] == 'binary' else \
                   [(child['content'].strip().casefold(), prompt(child).strip().casefold()) for child in expand([row])]
            for key in keys:
                if key in examples:
                    raise ValueError(f'{location}: duplicate input (possibly conflicting label)')
                examples.add(key)
            rows.append(row)
    if not rows:
        raise ValueError('No data found')
    return rows


def data_paths(args):
    base = [Path(p) for p in args.data] if getattr(args, 'data', None) else [ROOT / 'data/seed.jsonl', ROOT / 'data/plain-questions.jsonl']
    return [*base, *[Path(p) for p in args.extra_data]]


def metrics(rows, probabilities):
    import numpy as np
    y = np.array([r['label'] for r in rows])
    p = np.asarray(probabilities).clip(1e-7, 1 - 1e-7)
    predicted = p >= .5
    recall = [float((predicted[y == v] == v).mean()) for v in (0, 1) if (y == v).any()]
    return {'count': len(rows), 'accuracy': float((predicted == y).mean()),
            'balanced_accuracy': sum(recall) / len(recall),
            'brier': float(((p-y)**2).mean()),
            'log_loss': float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean()),
            'by_family': {family: {'count': sum(r['task_family'] == family for r in rows),
                          'accuracy': float((predicted[[r['task_family'] == family for r in rows]] == y[[r['task_family'] == family for r in rows]]).mean())}
                          for family in sorted({r['task_family'] for r in rows})}}


def encode(tokenizer, rows, template=1):
    questions, contents = [prompt(r) for r in rows], [r['content'] for r in rows]
    first, second = (contents, questions) if template == 2 else (questions, contents)
    padding = {'padding': 'max_length', 'max_length': MAX_TOKENS} if FIXED_SHAPES else {'padding': True}
    result = tokenizer(first, second, truncation=False, return_token_type_ids=True, return_tensors='pt', **padding)
    if result['input_ids'].shape[1] > MAX_TOKENS:
        raise ValueError(f'Input exceeds {MAX_TOKENS} tokens; shorten data or explicitly revise model contract')
    return result


def model_inputs(model, batch):
    """Drop segment ids for encoders whose forward() has no token_type_ids argument (ModernBERT)."""
    if 'token_type_ids' not in inspect.signature(model.forward).parameters:
        batch = {k: v for k, v in batch.items() if k != 'token_type_ids'}
    return batch


def predict(model, tokenizer, rows, device):
    import torch
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            batch = {k:v.to(device) for k,v in encode(tokenizer, rows[start:start+16], getattr(model.config, 'decisionmodel_template_version', 1)).items()}
            result.extend(model(**model_inputs(model, batch)).logits.softmax(-1)[:,1].cpu().tolist())
    if device == 'mps':
        torch.mps.empty_cache()
    return result


def train(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    paths = data_paths(args)
    rows = expand(read_data(paths))
    train_rows = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    if not train_rows or not validation:
        raise ValueError('Both train and validation data required')
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise ValueError('Output exists: choose a new run directory or --resume')
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    source = args.start or args.base
    source_options = {} if args.start else {'revision': args.revision}
    tokenizer = AutoTokenizer.from_pretrained(source, **source_options)
    if args.nli_head and not args.start:
        model = AutoModelForSequenceClassification.from_pretrained(source, attn_implementation='eager', **source_options)
        # Reuse the NLI head: yes starts as entailment, no as the average of the non-entailment classes.
        labels = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
        if 'entailment' not in labels or not ({'contradiction', 'neutral'} <= set(labels) or 'not_entailment' in labels):
            raise ValueError('NLI initialization requires an entailment/contradiction/neutral or entailment/not_entailment head')
        no_classes = [labels[n] for n in ('contradiction', 'neutral', 'not_entailment') if n in labels]
        holder, attribute = (model.classifier, 'out_proj') if hasattr(model.classifier, 'out_proj') else (model, 'classifier')
        old = getattr(holder, attribute)
        head = torch.nn.Linear(old.in_features, 2)
        with torch.no_grad():
            head.weight[0].copy_(old.weight[no_classes].mean(0))
            head.bias[0].copy_(old.bias[no_classes].mean())
            head.weight[1].copy_(old.weight[labels['entailment']]); head.bias[1].copy_(old.bias[labels['entailment']])
        setattr(holder, attribute, head)
        model.num_labels = model.config.num_labels = 2
    else:
        model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=2, attn_implementation='eager', **source_options)
    template = getattr(model.config, 'decisionmodel_template_version', args.template)
    model.config.decisionmodel_template_version = template
    parent_recipe = {}
    if args.start and (Path(args.start).parent/'config.json').is_file():
        parent_recipe = json.loads((Path(args.start).parent/'config.json').read_text())
    foundation_base = getattr(model.config, 'decisionmodel_base', parent_recipe.get('base', args.base))
    foundation_revision = getattr(model.config, 'decisionmodel_revision', parent_recipe.get('revision', args.revision))
    model.config.decisionmodel_base = foundation_base
    model.config.decisionmodel_revision = foundation_revision
    global FIXED_SHAPES
    FIXED_SHAPES = args.fixed_shapes
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})  # recompute activations in backward; large models otherwise exhaust MPS memory
    model.to(device)
    model.config.id2label = {0:'NO', 1:'YES'}
    model.config.label2id = {'NO':0, 'YES':1}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    config = {'base':foundation_base,'revision':foundation_revision,'template_version':template,'nli_head':args.nli_head,'start':args.start,'seed':args.seed,'batch_size':args.batch_size,
              'learning_rate':args.learning_rate,'device':device,'max_tokens':MAX_TOKENS,
              'data':{str(p):digest(p) for p in paths}, 'parameter_count':sum(p.numel() for p in model.parameters()),
              'torch':torch.__version__, 'epochs':args.epochs,
              'pipeline_sha256':digest(__file__), 'uv_lock_sha256':digest(ROOT/'uv.lock'),
              'start_sha256':digest(Path(args.start)/'model.safetensors') if args.start else None}
    first_epoch, best = 0, float('inf')
    history = []
    if args.resume:
        checkpoint = torch.load(output/'resume.pt', map_location='cpu', weights_only=False)
        if checkpoint['config'] != config:
            raise ValueError('Resume configuration/data differ from saved run')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for optimizer_state in optimizer.state.values():
            for k,v in optimizer_state.items():
                if isinstance(v, torch.Tensor): optimizer_state[k] = v.to(device)
        first_epoch, best, history = checkpoint['epoch'], checkpoint['best'], checkpoint['history']
        random.setstate(checkpoint['random_state'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if device == 'mps': torch.mps.set_rng_state(checkpoint['device_rng'])
        if device == 'cuda': torch.cuda.set_rng_state(checkpoint['device_rng'])
    else:
        baseline = metrics(validation, predict(model, tokenizer, validation, device))
        (output/'initial-validation.json').write_text(json.dumps(baseline, indent=2)+'\n')
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    start_time = time.monotonic()
    for epoch in range(first_epoch, args.epochs):
        model.train()
        order = list(range(len(train_rows)))
        random.shuffle(order)
        loss_sum = 0
        for offset in range(0, len(order), args.batch_size):
            batch_rows = [train_rows[i] for i in order[offset:offset+args.batch_size]]
            batch = {k:v.to(device) for k,v in encode(tokenizer, batch_rows, template).items()}
            labels = torch.tensor([r['label'] for r in batch_rows], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**model_inputs(model, batch), labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_sum += loss.item()*len(batch_rows)
            if device == 'mps' and (offset // args.batch_size) % 5 == 0:
                torch.mps.empty_cache()  # variable sequence lengths otherwise grow the MPS cache until it fails
        validation_probabilities = predict(model, tokenizer, validation, device)
        score = metrics(validation, validation_probabilities)
        choice_items = [(r, p) for r, p in zip(validation, validation_probabilities) if r.get('choice_of')]
        if choice_items:
            crows, cprobs = zip(*choice_items)
            extra = choice_metrics(list(crows), list(cprobs))
            score['choice_accuracy'], score['choice_count'] = extra['choice_accuracy'], extra['choice_count']
        train_score = metrics(train_rows, predict(model, tokenizer, train_rows, device))
        result = {'epoch':epoch+1, 'train_loss':loss_sum/len(train_rows), 'training':train_score, 'validation':score, 'elapsed_seconds':time.monotonic()-start_time}
        history.append(result)
        print(json.dumps(result), flush=True)
        if score['log_loss'] < best:
            best = score['log_loss']
            model.save_pretrained(output/'best')
            tokenizer.save_pretrained(output/'best')
        checkpoint = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,
                      'best':best,'config':config,'history':history,'random_state':random.getstate(),
                      'torch_rng':torch.get_rng_state(), 'device_rng':torch.mps.get_rng_state() if device=='mps' else torch.cuda.get_rng_state() if device=='cuda' else None}
        torch.save(checkpoint, output/'resume.tmp')
        (output/'resume.tmp').replace(output/'resume.pt')
        (output/'history.json').write_text(json.dumps(history, indent=2)+'\n')


def export(args):
    import numpy as np
    import torch
    import onnxruntime as ort
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(4)
    source, output = Path(args.checkpoint), Path(args.output)
    if output.exists(): raise ValueError('Export destination exists; choose a new bundle directory')
    output.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source, attn_implementation='eager').eval()
    tokenizer.backend_tokenizer.no_padding()
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.backend_tokenizer.save(str(output/'tokenizer.json'))
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = model
        def forward(self, input_ids, attention_mask, token_type_ids):
            if getattr(model.config, 'type_vocab_size', 0) > 0 and 'token_type_ids' in inspect.signature(model.forward).parameters:
                return self.inner(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
            # Architectures without segment embeddings (DeBERTa v3) ignore token_type_ids; keep the input in the
            # graph with a zero contribution so every runtime can feed the same three tensors.
            logits = self.inner(input_ids=input_ids, attention_mask=attention_mask).logits
            return logits + token_type_ids[:, :1].to(logits.dtype) * 0
    template = getattr(model.config, 'decisionmodel_template_version', 1)
    sample = encode(tokenizer, [{'question':'Is this asking for a refund?', 'content':'Please return my payment.'}], template)
    names = ['input_ids','attention_mask','token_type_ids']
    with torch.no_grad():
        torch.onnx.export(Wrapper().eval(), tuple(sample[n] for n in names), str(output/'model.onnx'),
            input_names=names, output_names=['logits'], opset_version=17, dynamo=False,
            dynamic_axes={**{n:{0:'batch',1:'sequence'} for n in names}, 'logits':{0:'batch'}})
    session = ort.InferenceSession(str(output/'model.onnx'), providers=['CPUExecutionProvider'])
    actual = session.run(None,{n:sample[n].numpy() for n in names})[0]
    with torch.no_grad(): expected = model(**model_inputs(model, sample)).logits.numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    manifest = {'format_version':1,'template_version':template,'choice_template_version':1,'model_id':args.model_id,'max_tokens':MAX_TOKENS,
                'base_model':BASE,'base_revision':REVISION,'calibration':'none; experimental uncalibrated softmax',
                'checkpoint_sha256':digest(source/'model.safetensors'),
                'sha256':{n:digest(output/n) for n in ('model.onnx','tokenizer.json')}}
    config_path = source.parent/'config.json'
    if config_path.is_file():
        manifest['training'] = json.loads(config_path.read_text())
        manifest['base_model'] = manifest['training']['base']
        manifest['base_revision'] = manifest['training']['revision']
    manifest['pipeline_sha256'] = digest(__file__)
    manifest['uv_lock_sha256'] = digest(ROOT/'uv.lock')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Exported {output}; single-fixture max logit error {np.max(np.abs(actual-expected)):.8g}')


def evaluate(args):
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    rows = [r for r in read_data(data_paths(args)) if r['split'] == args.split]
    if not rows: raise ValueError('Empty evaluation split')
    binary_rows = [r for r in rows if r['label_type'] == 'binary']
    choice_rows = [r for r in rows if r['label_type'] == 'choice']
    if args.command == 'calibrate' and not binary_rows:
        raise ValueError('Empty binary calibration split')
    bundle = Path(args.bundle)
    manifest = json.loads((bundle/'manifest.json').read_text())
    temperature = 1.0 if args.command == 'calibrate' else manifest.get('temperature', 1.0)
    tokenizer = Tokenizer.from_file(str(bundle/'tokenizer.json'))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(bundle/'model.onnx'), options, providers=['CPUExecutionProvider'])

    def raw_logits(content, text):
        encoded = tokenizer.encode(content, text) if manifest['template_version'] == 2 else tokenizer.encode(text, content)
        if len(encoded.ids)>MAX_TOKENS: raise ValueError('Overlength evaluation input')
        feed = {name:np.array([values],dtype=np.int64) for name,values in [('input_ids',encoded.ids),('attention_mask',encoded.attention_mask),('token_type_ids',encoded.type_ids)]}
        return session.run(None,feed)[0][0].astype(float)

    probabilities, timings = [], []
    for row in binary_rows:
        start = time.perf_counter()
        logits = raw_logits(row['content'], prompt(row)) / temperature
        p = np.exp(logits-logits.max()); p /= p.sum()
        probabilities.append(float(p[1])); timings.append(time.perf_counter()-start)
    if args.command == 'calibrate':
        p = np.asarray(probabilities).clip(1e-7, 1-1e-7)
        differences = np.log(p/(1-p))
        temperatures = np.exp(np.linspace(np.log(.1), np.log(10), 401))
        labels = np.array([r['label'] for r in binary_rows])
        def loss(t):
            logits = differences/t
            return float((np.logaddexp(0, logits)-labels*logits).mean())
        temperature = float(min(temperatures, key=loss))
        manifest.update({'temperature':temperature, 'calibration':'temperature scaling on synthetic calibration split; not a reliability guarantee',
                         'calibration_data':{str(p):digest(p) for p in data_paths(args)}})
        (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        probabilities = (1/(1+np.exp(-differences/temperature))).tolist()
    # Choice probabilities reuse this same fitted/loaded temperature: z per option = (logit_yes-logit_no)/temperature, softmax over a record's options.
    choice_children = expand(choice_rows)
    choice_probabilities = []
    for row in choice_children:
        logits = raw_logits(row['content'], prompt(row)) / temperature
        p = np.exp(logits-logits.max()); p /= p.sum()
        choice_probabilities.append(float(p[1]))
    result = metrics(binary_rows, probabilities) if binary_rows else {'count': 0}
    if binary_rows:
        result.update({'majority_accuracy':max(sum(r['label']==v for r in binary_rows) for v in (0,1))/len(binary_rows),
                       'p50_ms':float(np.percentile(timings,50)*1000),'p95_ms':float(np.percentile(timings,95)*1000),
                       'predictions':[{'id':r['id'],'label':r['label'],'p_yes':p} for r,p in zip(binary_rows,probabilities)]})
    result.update(choice_metrics(choice_children, choice_probabilities))
    result.update({'split':args.split, 'temperature':temperature, 'bundle':str(bundle),'model_sha256':digest(bundle/'model.onnx'),
                   'data_sha256':{str(p):digest(p) for p in data_paths(args)},
                   'constant_half_brier':0.25, 'constant_half_log_loss':float(np.log(2)),
                   'note':'Synthetic unreviewed pilot; these timings are not the 1000-call release benchmark.'})
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('predictions','choice_predictions')},indent=2))


def compress(args):
    """Store float32 weights as float16 with a Cast back to float32; compute stays float32, so predictions barely move."""
    import numpy as np
    import onnx
    from onnx import TensorProto, numpy_helper
    source, output = Path(args.bundle), Path(args.output)
    if output.exists(): raise ValueError('Compressed output already exists')
    output.mkdir(parents=True)
    model = onnx.load(str(source/'model.onnx'))
    graph = model.graph
    kept, halved, casts = [], [], []
    for initializer in graph.initializer:
        if initializer.data_type != TensorProto.FLOAT:
            kept.append(initializer); continue
        name = initializer.name + '.fp16'
        halved.append(numpy_helper.from_array(numpy_helper.to_array(initializer).astype(np.float16), name))
        casts.append(onnx.helper.make_node('Cast', [name], [initializer.name], to=TensorProto.FLOAT, name='CastToFp32__' + initializer.name))
    del graph.initializer[:]; graph.initializer.extend(kept + halved)
    nodes = casts + list(graph.node)
    del graph.node[:]; graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(output/'model.onnx'))
    shutil.copy2(source/'tokenizer.json', output/'tokenizer.json')
    manifest = json.loads((source/'manifest.json').read_text())
    manifest.update({'model_id':args.model_id,
                     'compression':{'method':'float16 weight storage with Cast to float32 at load; float32 compute','converted_initializers':len(halved)},
                     'float_model_sha256':digest(source/'model.onnx')})
    manifest['sha256'] = {n:digest(output/n) for n in ('model.onnx','tokenizer.json')}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Compressed weights: {(output/"model.onnx").stat().st_size} bytes ({len(halved)} tensors stored as float16); temperature kept, re-run calibrate and evaluate to confirm')


def quantize(args):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    source, output = Path(args.bundle), Path(args.output)
    if output.exists(): raise ValueError('Quantized output already exists')
    output.mkdir(parents=True)
    import onnx
    operators=['MatMul','Gemm'] + (['Gather'] if args.quantize_embeddings else [])
    graph = onnx.load(str(source/'model.onnx'), load_external_data=False).graph
    kept = [n.name for n in graph.node if n.op_type in operators and any(f in n.name for f in args.keep_float)]
    nodes = [n.name for n in graph.node if n.op_type in operators and n.name not in kept]
    quantize_dynamic(str(source/'model.onnx'),str(output/'model.onnx'),
                     op_types_to_quantize=operators,nodes_to_quantize=nodes,weight_type=QuantType.QInt8,
                     per_channel=True,extra_options={'MatMulConstBOnly':True})
    shutil.copy2(source/'tokenizer.json',output/'tokenizer.json')
    manifest=json.loads((source/'manifest.json').read_text())
    manifest.update({'model_id':args.model_id,'temperature':1.,'calibration':'unfitted after quantization',
                     'quantization':{'runtime':'ONNX Runtime','weight_type':'QInt8','operators':operators,'per_channel':True,'kept_float':kept},
                     'float_model_sha256':digest(source/'model.onnx')})
    manifest.pop('calibration_data',None)
    manifest['sha256']={n:digest(output/n) for n in ('model.onnx','tokenizer.json')}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Quantized weights: {(output/"model.onnx").stat().st_size} bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    valid = sub.add_parser('validate'); valid.add_argument('--extra-data',action='append',default=[])
    tr = sub.add_parser('train')
    tr.add_argument('--output',required=True); tr.add_argument('--extra-data',action='append',default=[])
    tr.add_argument('--epochs',type=int,default=12); tr.add_argument('--batch-size',type=int,default=16)
    tr.add_argument('--learning-rate',type=float,default=3e-5); tr.add_argument('--seed',type=int,default=42)
    tr.add_argument('--device',choices=['cpu','mps','cuda']); tr.add_argument('--start'); tr.add_argument('--resume',action='store_true')
    tr.add_argument('--base',default=BASE); tr.add_argument('--revision',default=REVISION)
    tr.add_argument('--nli-head',action='store_true'); tr.add_argument('--template',type=int,choices=[1,2],default=1)
    tr.add_argument('--gradient-checkpointing',action='store_true',help='Recompute activations during backward to cut peak memory')
    tr.add_argument('--fixed-shapes',action='store_true',help='Pad every batch to the maximum token count so GPU memory blocks are reused')
    ex = sub.add_parser('export'); ex.add_argument('--checkpoint',required=True); ex.add_argument('--output',required=True); ex.add_argument('--model-id',required=True)
    ev = sub.add_parser('evaluate'); ev.add_argument('--bundle',required=True); ev.add_argument('--split',choices=['train','validation','calibration','test'],default='test'); ev.add_argument('--output',required=True); ev.add_argument('--extra-data',action='append',default=[])
    cal = sub.add_parser('calibrate'); cal.add_argument('--bundle',required=True); cal.add_argument('--output',required=True); cal.add_argument('--extra-data',action='append',default=[]); cal.set_defaults(split='calibration')
    quant = sub.add_parser('quantize'); quant.add_argument('--bundle',required=True); quant.add_argument('--output',required=True); quant.add_argument('--model-id',required=True)
    quant.add_argument('--quantize-embeddings',action='store_true',help='Also compress embedding tables; can significantly change predictions')
    quant.add_argument('--keep-float',action='append',default=[],metavar='FRAGMENT',help='Leave nodes whose name contains this text in float; repeatable')
    comp = sub.add_parser('compress'); comp.add_argument('--bundle',required=True); comp.add_argument('--output',required=True); comp.add_argument('--model-id',required=True)
    for command in (valid,tr,ev,cal): command.add_argument('--data',action='append',help='Use these JSONL files instead of the default seed files')
    args = parser.parse_args()
    if args.command=='validate':
        from collections import Counter
        rows = read_data(data_paths(args))
        counts = Counter((r['split'], r['label_type']) for r in rows)
        print(json.dumps({f'{split}/{label_type}':n for (split,label_type),n in sorted(counts.items())}))
    elif args.command=='train': train(args)
    elif args.command=='export': export(args)
    elif args.command=='quantize': quantize(args)
    elif args.command=='compress': compress(args)
    else: evaluate(args)

if __name__=='__main__': main()
