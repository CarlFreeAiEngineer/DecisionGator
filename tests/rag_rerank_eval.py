#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["datasets>=2.20", "rank-bm25>=0.2.2", "sentence-transformers>=3", "numpy>=2"]
# ///
"""Can DecisionGator rerank retrieved chunks for retrieval-augmented generation?

For each question, a fixed set of candidate passages is scored and sorted, and
the ranking is compared with human relevance labels. Candidates come from two
public sets, downloaded at run time and never committed:

- MS MARCO v1.1 validation: web search questions, about ten Bing passages each,
  with `is_selected` marking passages an annotator used to write the answer.
  Unselected passages can still be relevant, so scores here are pessimistic.
- WikiQA test: questions with the sentences of one Wikipedia summary, labeled
  as answering the question or not. Short candidates, clean labels.

Scorers: random order, BM25 keyword ranking, DecisionGator is_yes_p with two
question phrasings, DecisionGator choose_p with passages as options (the
question as content), and two dedicated rerankers: ms-marco-MiniLM-L-6-v2 (22M
parameters, trained on MS MARCO, so on that set an upper reference, not a fair
rival) and bge-reranker-base (278M parameters, general purpose).

Usage: uv run tests/rag_rerank_eval.py [--queries 150] [--output reports/rag-rerank]
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import decisiongator as dg  # noqa: E402

PHRASINGS = {
    'dg_answers': 'Does this passage answer the question "{q}"?',
    'dg_relevant': 'Is this passage relevant to the question "{q}"?',
}


def load_msmarco(n, seed):
    from datasets import load_dataset
    rows = load_dataset('microsoft/ms_marco', 'v1.1', split='validation')
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    out = []
    for i in order:
        r = rows[i]
        labels = list(r['passages']['is_selected'])
        if 0 < sum(labels) < len(labels) and r['answers'] and r['answers'][0] != 'No Answer Present.':
            out.append({'id': f'msmarco-{r["query_id"]}', 'question': r['query'],
                        'passages': list(r['passages']['passage_text']), 'labels': labels})
        if len(out) == n:
            break
    return out


def load_wikiqa(n, seed):
    from datasets import load_dataset
    rows = load_dataset('microsoft/wiki_qa', split='test')
    groups = {}
    for r in rows:
        g = groups.setdefault(r['question_id'], {'id': f'wikiqa-{r["question_id"]}', 'question': r['question'],
                                                 'passages': [], 'labels': []})
        g['passages'].append(r['answer'])
        g['labels'].append(int(r['label']))
    usable = [g for g in groups.values() if 0 < sum(g['labels']) < len(g['labels'])]
    random.Random(seed).shuffle(usable)
    return usable[:n]


def metrics(scores, labels):
    """Ranking metrics for one question; ties broken by original order."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranked = [labels[i] for i in order]
    first = ranked.index(1) + 1
    dcg = sum(l / np.log2(k + 2) for k, l in enumerate(ranked[:10]))
    ideal = sum(l / np.log2(k + 2) for k, l in enumerate(sorted(labels, reverse=True)[:10]))
    return {'mrr': 1 / first, 'p_at_1': float(ranked[0] == 1), 'ndcg10': dcg / ideal,
            'recall_at_3': sum(ranked[:3]) / sum(labels)}


def auc(pos, neg):
    """Probability that a relevant passage outscores an irrelevant one, across all questions."""
    if not pos or not neg:
        return None
    pos, neg = np.array(pos), np.array(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def threshold_stats(pairs, t=0.5):
    """How well a fixed cutoff separates relevant from irrelevant (only for probability scorers)."""
    tp = sum(1 for s, l in pairs if s >= t and l)
    fp = sum(1 for s, l in pairs if s >= t and not l)
    fn = sum(1 for s, l in pairs if s < t and l)
    empty = None
    return {'threshold': t, 'precision': tp / (tp + fp) if tp + fp else empty,
            'recall': tp / (tp + fn) if tp + fn else empty}


def score_set(name, questions, scorers):
    results = {}
    for key, fn in scorers.items():
        started, per_q, pairs, errors = time.time(), [], [], 0
        for q in questions:
            try:
                s = fn(q)
            except dg.DecisionGatorError as e:
                errors += 1
                print(f'  {key} {q["id"]}: {e}', file=sys.stderr)
                continue
            q.setdefault('scores', {})[key] = s
            per_q.append(metrics(s, q['labels']))
            pairs += list(zip(s, q['labels']))
        agg = {m: float(np.mean([p[m] for p in per_q])) for m in per_q[0]}
        agg['questions'] = len(per_q)
        agg['errors'] = errors
        agg['auc'] = auc([s for s, l in pairs if l], [s for s, l in pairs if not l])
        if key.startswith('dg_'):
            agg['cutoff'] = threshold_stats(pairs)
        agg['seconds'] = round(time.time() - started, 1)
        results[key] = agg
        print(f'{name} {key}: MRR {agg["mrr"]:.3f} P@1 {agg["p_at_1"]:.3f} nDCG@10 {agg["ndcg10"]:.3f} '
              f'AUC {agg["auc"]:.3f} ({agg["seconds"]}s, {errors} errors)', flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', type=int, default=150)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--output', default=str(ROOT / 'reports' / 'rag-rerank'))
    ap.add_argument('--skip-choose', action='store_true')
    args = ap.parse_args()

    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    bge = CrossEncoder('BAAI/bge-reranker-base')
    words = lambda t: re.findall(r'\w+', t.lower())

    def bm25(q):
        return list(BM25Okapi([words(p) for p in q['passages']]).get_scores(words(q['question'])))

    def dg_phrasing(template):
        return lambda q: [dg.is_yes_p(p, template.format(q=q['question'])) for p in q['passages']]

    def dg_choose(q):
        ranked = dg.choose_p(q['question'], 'Which passage answers this question?', q['passages'])
        scores = [0.0] * len(q['passages'])
        for i, p in ranked:
            scores[i] = p
        return scores

    rng = random.Random(args.seed)
    scorers = {'random': lambda q: [rng.random() for _ in q['passages']], 'bm25': bm25,
               'minilm_reranker': lambda q: [float(x) for x in reranker.predict([(q['question'], p) for p in q['passages']])],
               'bge_reranker': lambda q: [float(x) for x in bge.predict([(q['question'], p) for p in q['passages']])]}
    scorers.update({k: dg_phrasing(t) for k, t in PHRASINGS.items()})
    if not args.skip_choose:
        scorers['dg_choose'] = dg_choose

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    session = dg.Session.load(dg._bundled_directory())
    report = {'model': session.metadata.get('model_id'), 'phrasings': PHRASINGS, 'queries_per_set': args.queries,
              'seed': args.seed, 'sets': {}}
    session.close()
    for name, loader in (('wikiqa', load_wikiqa), ('msmarco', load_msmarco)):
        questions = loader(args.queries, args.seed)
        report['sets'][name] = score_set(name, questions, scorers)
        # Per-passage scores hold dataset text, so they stay in the ignored tmp/ folder.
        scratch = ROOT / 'tmp' / 'rag-rerank'
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / f'{name}-scores.json').write_text(json.dumps(questions, indent=1))
        (out / 'summary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
