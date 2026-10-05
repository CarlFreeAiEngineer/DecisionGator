# DecisionGator gave a wrong answer. What now?

Good: a wrong answer is something you can fix yourself. There are two kinds of fix. Try the quick one first; it often works and needs no training.

## 1. Quick fix: say more in your call

The model only knows what you tell it. Before retraining, try:

- **Describe each option.** `["HR", "SALES"]` makes the model guess what each one covers. `["HR: jobs, employees and former employees, paychecks", "Sales: learning about our products before buying"]` tells it. In one test, descriptions made the model much more confident on right answers.
- **Add criteria** to a yes/no question, spelling out what counts as yes and what counts as no.
- **Use the probability.** Call `is_yes_p` or `choose_p` and send answers below, say, 0.6 to a person instead of acting on them. The model is usually less sure when it is wrong.

## 2. Real fix: teach it, and retrain

### Add your example to `data/corrections.jsonl`

That one file is the only one you edit. One example per line:

```text
{"text": "i am a former employee. when will my last check arrive?", "question": "Is this an HR matter?", "answer": "yes", "why": "Pay owed to former staff is HR."}
{"text": "Do you have an ai product?", "question": "Which department should handle this message?", "options": ["BILLING", "TECH SUPPORT", "SALES", "HR"], "answer": "SALES"}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `text` | yes | The content DecisionGator read |
| `question` | yes | The question you asked |
| `answer` | yes | `"yes"` or `"no"`, or, with `options`, the right option exactly as written |
| `options` | for multiple choice | The list of options you passed |
| `criteria` | no | `{"yes": "...", "no": "..."}` if your call used criteria |
| `why` | no | One sentence on why that is the right answer, for reviewers |

Tips:

- One example teaches little. Add a few varied ones: the same idea in different words, and a near miss whose answer is the other way.
- Use only text you have the right to share. Never paste real customer data; write a similar made-up message instead.
- Check your file without training: `uv run training/retrain.py --check-only`. It explains any mistake, and refuses an example that copies one of the project's test cases, since training on a test would make its score meaningless.

You can edit the file on GitHub itself: fork the repository, open `data/corrections.jsonl`, and use the pencil button.

### Retrain on Google Colab

[Open the retraining notebook in Colab](https://colab.research.google.com/github/CarlFreeAiEngineer/DecisionGator/blob/main/colab/retrain.ipynb). Choose an A100 GPU (Runtime → Change runtime type; needs a paid Colab plan), point the notebook at your fork if you made one, and run all cells. It takes about half an hour and ends with a download of `decisiongator-model.zip`.

Or, on your own Linux machine with an NVIDIA GPU of 40 GB or more:

```text
git clone https://github.com/CarlFreeAiEngineer/DecisionGator.git
cd DecisionGator
uv run training/retrain.py
```

Either way the script trains with exactly the recipe of the released model, adds your corrections, compresses and calibrates the result the same way, and prints a table: every held-out test score next to the released model's, and how the old and new model answer each of your corrections. Getting your corrections right shows the model learned them; the held-out tests show whether it got worse anywhere else. Check both.

### Use your new model

Unzip `decisiongator-model.zip` into your DecisionGator bundle folder (for example `released/windows-x64/`, or the `decisiongator` folder next to your program), replacing `model.onnx`, `tokenizer.json` and `manifest.json`. Keep a copy of the old files to switch back. Programs that load a bundle folder use the new model straight away: C, Rust, Go, and C# with the normal bundle. In Python, load the folder explicitly with `Session.load(path)`.

Packages that carry the model inside need rebuilding from the new bundle folder: the one-file library (`uv run code/build_standalone.py --bundle FOLDER`, see [code/standalone](code/standalone/README.md)), the Java JAR ([java/README.md](java/README.md)), the Node.js package (`DECISIONGATOR_BUNDLE=FOLDER npm run build`, see [javascript/README.md](javascript/README.md)), and the browser version, which stores the model in a different format.

## 3. Share it, so everyone's next release gets it

Open a pull request with your new lines in `data/corrections.jsonl`. You do not need to have retrained; the examples are the contribution. Reviewers check the label and the wording, and the next release is trained with them. Contributed examples are dedicated to the public domain (CC0-1.0), like the rest of the training data. Do not include model files in the pull request. More detail is in [CONTRIBUTING.md](CONTRIBUTING.md).
