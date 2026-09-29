# Reranking retrieved passages (RAG) with DecisionGator 0.4.1

Question: can DecisionGator sort a question's candidate passages so the relevant ones come first, as the reranking step of retrieval-augmented generation? Script: [tests/rag_rerank_eval.py](../../tests/rag_rerank_eval.py). Run: `uv run tests/rag_rerank_eval.py --queries 200` (about 50 minutes on the M1 Pro). Summary numbers: [summary.json](summary.json). Per-passage scores contain dataset text and stay in the ignored `tmp/rag-rerank/`.

## Data

200 random questions from each public set, each with at least one relevant and one irrelevant candidate:

- WikiQA test: the sentences of one Wikipedia summary, labeled as answering the question or not. About 9 short candidates per question, clean labels.
- MS MARCO v1.1 validation: about 8 web passages per question from Bing; `is_selected` marks the passages an annotator used for the answer. Unselected passages are often relevant too (duplicates, equally good answers), so every scorer is undercounted here.

## Results

MRR is 1 divided by the rank of the first relevant passage, averaged. Top-1 is how often the first passage is relevant.

| Scorer | WikiQA MRR | WikiQA top-1 | MS MARCO MRR | MS MARCO top-1 | Seconds per passage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random order | 0.37 | 17% | 0.36 | 14% | |
| BM25 keyword ranking | 0.62 | 45% | 0.43 | 20% | |
| ms-marco-MiniLM-L-6 reranker (22M) | 0.82 | 72% | 0.67 | 50% | 0.003 to 0.005 |
| bge-reranker-base (278M) | 0.84 | 76% | 0.68 | 52% | 0.008 to 0.014 |
| DG `is_yes_p`, "Does this passage answer the question ...?" | 0.83 | 75% | 0.53 | 33% | 0.12 to 0.25 |
| DG `is_yes_p`, "Is this passage relevant to the question ...?" | 0.78 | 66% | 0.51 | 30% | |
| DG `choose_p`, question as content, passages as options | 0.81 | 70% | 0.53 | 32% | |

Timings varied with other load on the machine (one `choose_p` pass took four times as long as another), so treat them as rough. One MS MARCO question was skipped by all DecisionGator scorers because a passage plus the question exceeded the 256-token limit.

With a fixed cutoff of 0.5, "Does this passage answer" marks most passages as answers: on WikiQA 31% of the passages above 0.5 are labeled relevant (81% of relevant ones pass), on MS MARCO 18% (83% pass). `choose_p` probabilities sum to one per question, so its cutoff numbers are not comparable.

## Reading

- Ranking works. On WikiQA, which none of these models was trained on as far as we know, DecisionGator matches the dedicated rerankers. The "answers the question" phrasing is best; `choose_p` adds nothing over it and cannot give an absolute score.
- The MS MARCO gap is partly unfair. MiniLM was trained on MS MARCO and bge-reranker's training data includes it, as far as we know. In 57 questions bge's top passage was labeled relevant and DecisionGator's was not. In a sample of five of them, three of DecisionGator's picks also answer the question (near-duplicates, other sources giving the same answer). Some are real misses: a passage about Napoleonic armies in general ranked above one with the numbers asked for.
- The probability is not usable as an absolute relevance cutoff. The model says yes to passages on the right topic that do not answer the question. This is the clearest thing training could fix, with pairs of on-topic passages that do and do not answer.
- It is 15 to 80 times slower than the rerankers and about 60 times larger than MiniLM. Reranking 20 chunks costs 3 to 5 seconds on the M1 Pro.

Verdict: not perfect, not hopeless. Ranking quality is competitive where the comparison is fair. Before training for this use, a fair test set that none of the rerankers has seen would settle the MS MARCO question, and the cost remains a reason to prefer a small dedicated reranker when speed matters. Training on "on topic but does not answer" passages would most likely make the probability usable as a cutoff.

## Possible follow-up: more training, same architecture

Not scheduled. If reranking becomes a supported use, improve it with training data only: keep the current model, input format, 256-token limit and API, and add no reranking-specific model, head or function. Callers keep using `is_yes_p(chunk, 'Does this passage answer the question "..."?')`.

1. Write a held-out test that none of the compared rerankers was trained on: questions with several passages on the same topic, only some of which answer. Score 0.4.x on it with this script before training.
2. Add training records in matched pairs: the same question with one on-topic passage that answers it (yes) and one that does not (no), across varied domains. Keep passages well under the token limit.
3. Continue training from the current checkpoint as in [the 0.4.2 report](../v4.2/README.md), then rerun all existing held-out tests to check that yes/no and choice accuracy do not drop.
4. Success means precision at the 0.5 cutoff rises well above today's 18 to 31% while ranking stays at least as good. Speed will not change; that would need a smaller model, which is out of scope here.
