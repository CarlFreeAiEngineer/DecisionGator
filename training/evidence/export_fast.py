"""Export a train_message.py checkpoint to ONNX with ModernBERT's local layers computed in bands.

ModernBERT uses full attention only every config.global_attn_every_n_layers layers; the others see a window of
config.local_attention tokens (a query at i sees keys j with |i - j| <= local_attention // 2). The plain eager
export computes full n-by-n attention with a mask in every layer. Here the local layers split the sequence into
blocks of half the window, and each block attends only to itself and its two neighbour blocks, with the same
window rule, so their cost grows with the length instead of its square. Global layers stay full attention.
The sequence is padded to a multiple of the block size inside the graph; the padding is masked.

Inputs and outputs match train_message.py's export: input_ids, attention_mask, marker_positions (int64)
-> logits [batch, n_markers, 3].

Export:            uv run python -m training.evidence.export_fast --checkpoint runs/evidence-m1/epoch-2 --output runs/evidence-m1/onnx-fast
Also 8-bit:        add --quantize (writes model.int8.onnx: dynamic QInt8, reduce_range)
Speed comparison:  add --bench (also exports the plain eager model to OUTPUT/full and times both)
Untrained model for development (train_message.py's model-building code on the base model, random 3-way head):
                   uv run python -m training.evidence.export_fast --make-dev-checkpoint --checkpoint tmp/dev-ckpt --output tmp/dev-onnx
"""
import argparse
import json
from pathlib import Path
import statistics
import time

from training.evidence import train as T
from training.evidence.train_message import message_logits


def load(checkpoint):
    import torch
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(checkpoint, attn_implementation='eager', torch_dtype=torch.float32).eval().cpu()


def make_dev_checkpoint(path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(T.BASE, revision=T.REVISION)
    args = argparse.Namespace(base=T.BASE, revision=T.REVISION)
    model = T.build_model(args, tokenizer)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def banded_module(model):
    """A torch module computing the same logits as message_logits(model, ...) with banded local attention."""
    import torch
    from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

    config = model.config
    window = config.local_attention // 2          # a query sees keys at distance <= window
    block = window                                # half the window: neighbour blocks cover every allowed key
    every = config.global_attn_every_n_layers
    pad_id = config.pad_token_id
    base = model.model
    global_rotary = next(l.attn.rotary_emb for i, l in enumerate(base.layers) if i % every == 0)
    local_rotary = next((l.attn.rotary_emb for i, l in enumerate(base.layers) if i % every != 0), None)
    for i, layer in enumerate(base.layers):
        assert (layer.attn.local_attention == (-1, -1)) == (i % every == 0)
        assert layer.attn.local_attention in ((-1, -1), (window, window))

    # Allowed (query offset in block, key offset in the 3-block band) pairs: key k in the band sits at relative
    # position k - block from the start of the query's block.
    q_off = torch.arange(block).unsqueeze(1)
    k_off = torch.arange(3 * block).unsqueeze(0) - block
    band_allowed = ((k_off - q_off).abs() <= window)  # [block, 3 * block]

    class Banded(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = model
            self.register_buffer('band_allowed', band_allowed, persistent=False)

        def forward(self, input_ids, attention_mask, marker_positions):
            b, n = input_ids.shape
            # Pad to a multiple of the block size; padded tokens are masked everywhere.
            padded = (n + block - 1) // block * block
            extra = padded - n
            input_ids = torch.cat([input_ids, torch.full((b, block), pad_id, dtype=input_ids.dtype)], 1)[:, :padded]
            mask = torch.cat([attention_mask.bool(), torch.zeros((b, block), dtype=torch.bool)], 1)[:, :padded]
            blocks = padded // block
            neg = torch.finfo(torch.float32).min
            global_bias = torch.zeros((b, 1, 1, padded)).masked_fill(~mask[:, None, None, :], neg)
            # Key validity for each block's band: [b, blocks, 3 * block]
            mask_p = torch.cat([torch.zeros((b, block), dtype=torch.bool), mask, torch.zeros((b, block), dtype=torch.bool)], 1)
            key_ok = torch.cat([mask_p[:, s * block:s * block + padded].reshape(b, blocks, block) for s in range(3)], 2)
            allowed = self.band_allowed[None, None, None] & key_ok[:, None, :, None, :]  # [b, 1, blocks, block, 3 * block]
            local_bias = torch.zeros(allowed.shape).masked_fill(~allowed, neg)

            hidden = base.embeddings(input_ids=input_ids)
            positions = torch.arange(padded).unsqueeze(0)
            rot_global = global_rotary(hidden, positions)
            rot_local = local_rotary(hidden, positions) if local_rotary is not None else None
            for i, layer in enumerate(base.layers):
                attn = layer.attn
                heads, dim = attn.num_heads, attn.head_dim
                qkv = attn.Wqkv(layer.attn_norm(hidden)).view(b, padded, 3, heads, dim)
                q, k, v = qkv.transpose(3, 1).unbind(dim=2)  # [b, heads, padded, dim]
                cos, sin = rot_global if i % every == 0 else rot_local
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                scale = dim ** -0.5
                if i % every == 0:
                    w = torch.matmul(q, k.transpose(2, 3)) * scale + global_bias
                    out = torch.matmul(torch.softmax(w, dim=-1), v)
                else:
                    qb = q.reshape(b, heads, blocks, block, dim)
                    zeros = torch.zeros((b, heads, block, dim))
                    kp = torch.cat([zeros, k, zeros], 2)
                    vp = torch.cat([zeros, v, zeros], 2)
                    kb = torch.cat([kp[:, :, s * block:s * block + padded].reshape(b, heads, blocks, block, dim) for s in range(3)], 3)
                    vb = torch.cat([vp[:, :, s * block:s * block + padded].reshape(b, heads, blocks, block, dim) for s in range(3)], 3)
                    w = torch.matmul(qb, kb.transpose(3, 4)) * scale + local_bias  # [b, heads, blocks, block, 3 * block]
                    out = torch.matmul(torch.softmax(w, dim=-1), vb).reshape(b, heads, padded, dim)
                out = out.transpose(1, 2).reshape(b, padded, heads * dim)
                hidden = hidden + attn.out_drop(attn.Wo(out))
                hidden = hidden + layer.mlp(layer.mlp_norm(hidden))
            hidden = base.final_norm(hidden)
            picked = hidden.gather(1, marker_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))
            return model.classifier(model.drop(model.head(picked)))

    return Banded().eval()


def plain_module(model):
    import torch

    class Plain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = model

        def forward(self, input_ids, attention_mask, marker_positions):
            return message_logits(self.inner, input_ids, attention_mask, marker_positions)

    return Plain().eval()


def random_input(n, markers, vocab, seed):
    """One message of n tokens: [CLS] random tokens [SEP], marker positions spread over the first half."""
    import torch
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, vocab - 1000, (1, n), generator=g)
    ids[0, 0], ids[0, -1] = 50281, 50282  # ModernBERT [CLS], [SEP]
    pos = torch.linspace(1, max(1, n // 2), markers).long().unsqueeze(0)
    return ids, torch.ones((1, n), dtype=torch.long), pos


def export_onnx(module, path, example):
    import torch
    with torch.no_grad():
        torch.onnx.export(module, example, str(path), input_names=['input_ids', 'attention_mask', 'marker_positions'],
                          output_names=['logits'], opset_version=17, dynamo=False,
                          dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}, 'attention_mask': {0: 'batch', 1: 'sequence'},
                                        'marker_positions': {0: 'batch', 1: 'markers'}, 'logits': {0: 'batch', 1: 'markers'}})


def session(path, threads):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def feed(ids, mask, pos):
    return {'input_ids': ids.numpy(), 'attention_mask': mask.numpy(), 'marker_positions': pos.numpy()}


def example_pair(vocab):
    """Two messages of different lengths and marker counts in one batch, the shorter padded."""
    import torch
    a, b = random_input(150, 2, vocab, 1), random_input(301, 5, vocab, 2)
    ids = torch.full((2, 301), 50283, dtype=torch.long)
    mask = torch.zeros((2, 301), dtype=torch.long)
    pos = torch.zeros((2, 5), dtype=torch.long)
    ids[0, :150], mask[0, :150], pos[0, :2] = a[0][0], 1, a[2][0]
    ids[1], mask[1], pos[1] = b[0][0], 1, b[2][0]
    return ids, mask, pos


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, help='folder saved by train_message.py (model and tokenizer)')
    parser.add_argument('--output', required=True, help='folder for model.onnx (and model.int8.onnx, full/)')
    parser.add_argument('--make-dev-checkpoint', action='store_true', help='first build an untrained checkpoint at --checkpoint')
    parser.add_argument('--quantize', action='store_true', help='also write model.int8.onnx (dynamic QInt8, reduce_range)')
    parser.add_argument('--bench', action='store_true', help='also export the plain eager model to OUTPUT/full and time all files')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--lengths', default='512,1000,2000,3500', help='token counts timed by --bench')
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import AutoTokenizer
    checkpoint, output = Path(args.checkpoint), Path(args.output)
    if args.make_dev_checkpoint:
        make_dev_checkpoint(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    model = load(checkpoint)
    vocab = model.config.vocab_size
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.backend_tokenizer.no_padding()
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.backend_tokenizer.save(str(output / 'tokenizer.json'))

    banded = banded_module(model)
    example = example_pair(vocab)
    export_onnx(banded, output / 'model.onnx', example)
    fast = session(output / 'model.onnx', args.threads)

    # Equivalence with the PyTorch model (default eager attention), one message at a time and one padded batch.
    report = {'checks': []}
    cases = [random_input(n, k, vocab, 10 + i) for i, (n, k) in enumerate([(300, 3), (1200, 6), (3500, 4), (777, 1), (2049, 9)])]
    cases.append(example)
    worst = 0.0
    for ids, mask, pos in cases:
        with torch.no_grad():
            expected = message_logits(model, ids, mask, pos).numpy()
            mine = banded(ids, mask, pos).numpy()
        actual = fast.run(None, feed(ids, mask, pos))[0]
        diff = float(np.abs(actual - expected).max())
        if ids.shape[0] > 1:  # padded rows of marker_positions are meaningless; compare only real markers
            diff = max(float(np.abs(actual[0, :2] - expected[0, :2]).max()), float(np.abs(actual[1] - expected[1]).max()))
        report['checks'].append({'shape': list(ids.shape), 'markers': pos.shape[1], 'onnx_vs_torch': diff,
                                 'banded_torch_vs_torch': float(np.abs(mine - expected).max())})
        worst = max(worst, diff)
        print(report['checks'][-1], flush=True)
    report['max_difference'] = worst
    if worst >= 1e-3:
        raise SystemExit(f'Banded export differs from the PyTorch model by {worst:.3g} (limit 1e-3)')

    if args.quantize or args.bench:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(output / 'model.onnx'), str(output / 'model.int8.onnx'), weight_type=QuantType.QInt8, reduce_range=True)
        q = session(output / 'model.int8.onnx', args.threads)
        qd = 0.0
        for ids, mask, pos in cases[:5]:
            qd = max(qd, float(np.abs(q.run(None, feed(ids, mask, pos))[0] - fast.run(None, feed(ids, mask, pos))[0]).max()))
        report['int8_max_difference_from_float'] = qd
        print('int8 max logit difference from float:', qd, flush=True)

    if args.bench:
        (output / 'full').mkdir(exist_ok=True)
        export_onnx(plain_module(model), output / 'full' / 'model.onnx', example)
        sessions = {'full_float32': session(output / 'full' / 'model.onnx', args.threads), 'banded_float32': fast,
                    'banded_int8': session(output / 'model.int8.onnx', args.threads)}
        report['seconds'] = {}
        for n in [int(x) for x in args.lengths.split(',')]:
            inputs = feed(*random_input(n, 4, vocab, n))
            row = {}
            for name, s in sessions.items():
                s.run(None, inputs)
                times = []
                for _ in range(3):
                    t = time.perf_counter(); s.run(None, inputs); times.append(time.perf_counter() - t)
                row[name] = round(statistics.median(times), 3)
            report['seconds'][n] = row
            print(n, row, flush=True)
    (output / 'export_fast.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Exported {output / "model.onnx"}; max logit difference {worst:.3g}', flush=True)


if __name__ == '__main__':
    main()
