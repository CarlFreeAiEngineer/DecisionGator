# Contributing to DecisionGator

The most valuable contribution is a wrong answer with the right answer attached. You do not need a GPU or machine-learning experience. This page is the short path; [the open training specification](specs/open-training.md) has the full rules.

## 1. Clone and get the prebuilt component

```text
git clone https://github.com/CarlFreeAiEngineer/DecisionGator.git
cd DecisionGator
uv run code/fetch_released.py --only macos-arm64    # or linux-x64, windows-x64, web, python, java, node
```

The prebuilt bundles are too large for GitHub, so they are downloaded from `https://62-84-178-253.sslip.io/DecisionGator/files/` and verified by checksum into `released/`. After that, [EXAMPLE_USAGE.md](EXAMPLE_USAGE.md) shows the calls in every language and `uv run examples/python_smoke.py` proves it works.

## 2. Record the wrong answer

Add one line per example to [data/corrections.jsonl](data/corrections.jsonl): the text, the question, and the right answer, plus the options for a multiple-choice question. [WRONG-ANSWER.md](WRONG-ANSWER.md) explains every field, with examples. Use only text you have the right to publish; never paste real customer data.

Check it:

```text
uv run training/retrain.py --check-only
```

## 3. Retrain (optional)

You do not need to retrain to contribute examples. To build and test a model with your corrections, open [the retraining notebook](https://colab.research.google.com/github/CarlFreeAiEngineer/DecisionGator/blob/main/colab/retrain.ipynb) on a paid Colab A100, or on a Linux machine with an NVIDIA GPU of 40 GB or more run:

```text
uv run training/retrain.py
```

It retrains with the released model's exact recipe plus your corrections, then reports every held-out test score next to the released model's. Its output is `model.onnx`, `tokenizer.json` and `manifest.json`, which replace the files in a bundle folder; see [WRONG-ANSWER.md](WRONG-ANSWER.md#use-your-new-model). The individual steps it runs (`decisiongator-train train`, `export`, `calibrate`, `evaluate`) are in [training/pipeline.py](training/pipeline.py) for anyone who wants to change the recipe.

You can stop here and ship your private variant. Nothing is uploaded anywhere.

## 4. Send the examples back

Open a pull request that adds your lines to `data/corrections.jsonl`, and say in the description where the examples came from and what they fix. Include the results table from `retrain.py` if you ran it. Do not include trained weights or bundles in the pull request; maintainers retrain from the merged data, run the release checks, and publish new bundles with `code/publish_released.py`.

Reviewers check that the text is yours to share, the question wording, the label, and whether the new examples overlap the evaluation sets. A correction that helps one family but hurts another is still welcome; just report both.

## Code changes

Code is Apache-2.0. Keep the C interface in `code/include/decisiongator.h` and every language wrapper in agreement; `tests/` and the per-language READMEs describe the checks each change must pass. Run the smoke tests for any language you touch before opening the pull request.
