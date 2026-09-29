# Model and training

## Starting approach

Fine-tune an existing language-understanding model. Training general language understanding from scratch is outside the initial project scope. A small output space reduces output work; it does not make understanding arbitrary questions trivial.

Start by comparing compact pretrained encoders that jointly read the content, question, and criteria and produce a binary classification score. A classification head is a small learned output layer that maps the model's representation to the decision. Prefer one forward evaluation without generated answer tokens.

The first size search should cover roughly 30–150 million total parameters, subject to available checkpoints and licenses. At one byte per parameter, 100 million parameters require roughly 100 MB for weights alone; runtime code, tokenizer, nonquantized tensors, and working memory are additional. These are planning estimates, not package sizes.

Use an existing natural language inference model as an initial baseline. Such models assess whether one text supports another. Their entailment, contradiction, and neutral scores are not automatically the required yes/no probability. In particular, dropping neutral and renormalizing the other two can turn lack of evidence into false certainty. Measure the baseline, then train the intended binary task explicitly.

Compare an instruction-aware classifier if an appropriate redistributable checkpoint is available. If compact encoders fail the unseen-question tests, compare a small decoder model with a classification head or constrained yes/no scoring. Account for the entire input-processing cost and calibrate its outputs too. Do not adopt a larger architecture until the smaller approach's failures are documented.

## Training environment

The inspected development device is an Apple M1 Pro with 32 GB unified memory. Start with PyTorch and its MPS backend for Apple GPU acceleration, after a small compatibility and memory test. Use CPU for unsupported operations only when explicitly recorded; silent fallback can hide severe performance problems.

Profile a short run before a full experiment: sequence length, batch size, peak memory, examples per second, and estimated total duration. Reduce batch size or use gradient accumulation when useful. Local fine-tuning is plausible at the proposed scale, not yet demonstrated.

Google Colab Pro is the fallback training environment. Record the actual allocated accelerator and memory; a subscription is not a guarantee of a particular GPU. Use the same versioned data, configuration, and checkpoint formats locally and in Colab. Save resumable checkpoints so a session ending does not discard the run.

Use `uv` for Python dependency management. A multi-file implementation gets `pyproject.toml` and a committed `uv.lock`; standalone scripts get PEP 723 dependency metadata and a `uv run --script` shebang. Pin model revisions and dependency versions. Record seeds, optimizer settings, preprocessing, hardware, and data hashes with each experiment.

## Data design

Each example records content, question, optional criteria, label, label provenance, source identifier, task family, and split. Keep uncertain or disputed annotations distinguishable from clear binary labels. Document whether any fractional target reflects annotator disagreement, observed frequency, or a teacher estimate; those meanings are not interchangeable.

Start with a small human-reviewed evaluation set across intent, textual evidence, policy conditions, and matching/duplicate judgments. Include positives, negatives, negation, quotations, mixed intent, implicit requests, absent evidence, contradictory evidence, numerical boundaries, and distracting instructions. Evaluate ordinary questions without criteria as well as questions with criteria.

Gather and license training data separately. Synthetic examples or a stronger local model may help enlarge it, but generated labels require quality checks and are not independent ground truth. Hosted teachers are optional research tools, not a runtime dependency or an assumed budget. Do not make copying Jev's outputs a prerequisite.

Split by source and task family before generating paraphrases or synthetic variations. Keep related documents, templates, and paraphrases in the same partition. Maintain separate training, model-selection validation, calibration, and final test sets. Include final tests with entire question families and source domains absent from training.

## Objective and calibration

Train the binary head with binary cross-entropy or an equivalent proper probabilistic loss. Use soft labels only when their meaning is justified. Track class balance and natural application prevalence; artificial balancing can change probability estimates even when classification accuracy improves.

Fit a simple calibration transform on a separate representative calibration partition. Temperature scaling is the starting candidate, not a guarantee. Evaluate calibration again after export and quantization, and fit the deployed model's transform without accessing the final test labels.

Choose architecture and training scale from measured accuracy, probability quality, memory, and latency. Do not prescribe reinforcement learning simply because Jev describes using it. Supervised learning and held-out calibration are the first experiment.

Passage reranking for retrieval-augmented generation was measured in [the reranking report](../reports/rag-rerank/README.md): ranking is competitive, but probabilities are too high for on-topic passages that do not answer. Any improvement there is planned as additional training data on the existing architecture; see that report's follow-up section.
