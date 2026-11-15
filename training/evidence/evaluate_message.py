"""Score a trained per-message model on the test split at another token limit (how much evidence the caller keeps).

    uv run -m training.evidence.evaluate_message --run runs/<name> --max-tokens 2048 --data ... [--relabel ...]

Reads the best checkpoint and temperature from the run, writes <run>/predictions-<max_tokens>.jsonl in the shape
score_real.py reads. Evidence is cut from the front, as in training.
"""
import argparse, json
from pathlib import Path

from training.evidence import train as T
from training.evidence import train_message as M


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--max-tokens', type=int, required=True)
    ap.add_argument('--data', action='append', required=True)
    ap.add_argument('--relabel', action='append', default=[])
    ap.add_argument('--tokens-per-batch', type=int, default=14000)
    a = ap.parse_args()
    run = Path(a.run)
    metrics = json.loads((run / 'metrics.json').read_text())
    checkpoint = run / f"epoch-{metrics['best_epoch']}"
    rows = T.load_data([Path(p) for p in a.data])
    if a.relabel:
        T.apply_relabels(rows, a.relabel)
    M.mark_scope(rows)
    messages = [m for m in M.build_messages(rows) if m['split'] == 'test']
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    M.encode_messages(tokenizer, messages, a.max_tokens)
    print('test messages', len(messages), 'evidence cut in', sum(m['cut_evidence'] for m in messages), flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    logits = M.predict_message_logits(model, messages, device, tokenizer.pad_token_id, a.tokens_per_batch, device == 'cuda')
    rows_t, logits_t = M.flatten(messages, logits)
    with open(run / f'predictions-{a.max_tokens}.jsonl', 'w', encoding='utf-8') as f:
        for r, p in zip(rows_t, torch.softmax(logits_t / metrics['temperature'], dim=-1).tolist()):
            f.write(json.dumps({'id': r['id'], 'split': 'test', 'source': r['source_file'], 'label': r['label'], 'kind': r.get('kind'),
                                'borderline': bool(r.get('borderline')), 'run': r.get('run'), 'question': r.get('question'),
                                'probs': [round(x, 4) for x in p]}) + '\n')
    print('done', flush=True)


if __name__ == '__main__':
    main()
