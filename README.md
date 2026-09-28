# DecisionGator

## Your software probably needs this.

## A Jev-like local AI inside a single .dll or .so

You’re building an application. A user types, “Any chance I could come in next Tuesday?” Your software needs to make one simple decision: is this person asking for an appointment?

Do you really have to get an API key, ship their POTENTIALLY PRIVATE text across the internet to a company they've never heard of, and pay for an API call just to answer that? Or integrate with Ollama or TensorFlow, manage a separate runtime, and turn a simple question into an infrastructure project? And most of the time, you end up BEGGING for JSON!

<img src="images/begging.jpg" alt="A developer on their knees begs a smug AI robot holding a coffee mug: Please. Just valid JSON." width="600">

```prompt
User text: "Any chance I could come in next Tuesday?"

Is this user asking for an appointment?

Please return ONLY valid JSON: {"appointment_requested": true}
Use false if the answer is no. An actual boolean, not the string "false".
No explanation. No Markdown. No triple-quoted code fences. No extra keys.
Do not say "Certainly!"; just skip to the JSON.

Please. I have a family. Just valid JSON... I'm begging you.
```

You're thinking: “I wish I could just load a normal library and call a normal function.”

**NOW YOU CAN!** In MANY languages and platforms; python example:

```python
from decisiongator import is_yes, choose

appointment_requested = is_yes(
    "Any chance I could come in next Tuesday?",
    "Is this person asking for an appointment?",
)   # True

teams = ["support", "billing", "sales"]
team = teams[choose(
    "Any chance I could get a discount if we buy 50 licenses?",
    "Which team should handle this message?",
    teams,
)]   # "sales"
```

**Don't take our word for it: [try it in your browser](https://62-84-178-253.sslip.io/DecisionGator/try.html).** Change the text or the question, press Run, and it answers on your own CPU. Nothing you type leaves the page.

**DecisionGator is a SOFTWARE COMPONENT, not an AI!** OK, there is a small trained model inside, but it is so small and insulated that you don't need to know about it. There is no chat, no prompt, no agent, and nothing to "talk to": it's a function that takes text and a question and returns a boolean. Ship it with your application and call a function. The same way your grandfather used software components: a library, some arguments, a return value. The kind of thing you did back in the 1990s.

**No Ollama. No llama.cpp. No TenserFlow, No agent harness. No network! Just a goddamn software component!**

**It's like a regex library.** Every language has one. You hand it a pattern and some text, and you get back a boolean. Nobody runs a regex server, nobody has a regex API key, and nobody thinks of it as AI. DecisionGator is the same kind of thing, except the pattern is a question in plain English: `is_yes(text, "Is this person asking for an appointment?")`. Reach for it on the day the regex stops working. Which, for human language input, is TODAY.

**The silent half of the brain.** In split-brain patients only one hemisphere can talk; the other still answers questions correctly, by pointing, and never explains itself ([CGP Grey explains](https://www.youtube.com/watch?v=wfYbgdo8e-8)). Language models are the talking half. DecisionGator is the silent half: it reads, decides, points at yes or no or one of your options, and says nothing else. Give your code both.

**If you know [JEV](https://www.jevai.org/)** from TypeSafe AI: the brain under the hood is similar, a small classifier that takes text plus a question and returns a typed answer with a probability instead of generating prose ([LangChain's write-up](https://www.langchain.com/blog/building-a-harness-with-jev)). But JEV is a paid API on someone else's server. DecisionGator is a library file in your build. For a working programmer that is a night-and-day difference: no account, no network, no bill, no one else's outage, and the model is yours to retrain.

## Languages and platforms

**An offline yes/no decision component for your software.** Ready-to-use interfaces for **Python, Java, TypeScript/JavaScript, C#, C, Rust, and Go** run on Apple silicon Macs, Windows x64, and Linux x64. The C interface also makes bindings possible for C++, Swift, Ruby, and other languages that can call C libraries.

| Platform             | Library     | Status                                  |
| -------------------- | ----------- | --------------------------------------- |
| Mac M1 and newer     | `.dylib`    | Experimental build, tested locally      |
| Windows x64          | `.dll`      | Built and tested on Windows 11          |
| Linux x64, glibc     | `.so`       | Built and tested on Omarchy Linux       |
| Desktop web browsers | WebAssembly | Tested in Chromium, Firefox, and WebKit |

**The same trained weights everywhere.** Native applications and browsers use the same decision data, tokenizer, and rules, with tested agreement between their answers.

The component runs on a CPU. Your application needs no separate runner or GPU setup. See [release bundles](released/README.md) for packaging details.

The same call, from whatever you already write in:

```java
Decisions.isYes(text, "Is the customer asking to cancel?")      // Java
```

```typescript
await isYes(text, "Is this reporting a service outage?");        // TypeScript / Node.js
```

```csharp
Decisions.IsYes(text, "Is the customer asking to cancel?")       // C#
```

```c
dg_is_yes(text, strlen(text), q, strlen(q), NULL, &yes);         /* C */
```

```rust
dg_is_yes(text.as_ptr(), text.len(), q.as_ptr(), q.len(), null(), &mut yes)  // Rust
```

```go
IsYes(text, "Is the customer asking to cancel?", nil, 0.5)       // Go, via cgo
```

**Even inside a web browser**, with the text never leaving the user's device:

```javascript
import { isYes } from "decisiongator/web";
```

Complete, runnable examples for every language, including probabilities, criteria, thresholds, and multiple-choice decisions, are in [EXAMPLE_USAGE.md](EXAMPLE_USAGE.md).

## Your answer. Your rules.

Supply different content and a yes/no question on each call. Optional criteria let you spell out what counts as yes or no. The first call loads the component and is slow; every call after that reuses it and takes about 300 milliseconds on a laptop CPU for a 256-token input and about 115 for a short one (measured p50 on an M1 Pro using its eight performance cores, see [the 0.4.1 report](reports/accuracy-v4.1.md)); the previous, far less accurate model took 70 milliseconds.

| Language                | Boolean                | Probability of yes      |
| ----------------------- | ---------------------- | ----------------------- |
| Python                  | `is_yes(...)`          | `is_yes_p(...)`         |
| Java                    | `Decisions.isYes(...)` | `Decisions.isYesP(...)` |
| JavaScript / TypeScript | `await isYes(...)`     | `await isYesP(...)`     |
| C#                      | `Decisions.IsYes(...)` | `Decisions.IsYesP(...)` |
| C                       | `dg_is_yes(...)`       | `dg_is_yes_p(...)`      |
| Rust (via the C ABI)    | `dg_is_yes(...)`       | `dg_is_yes_p(...)`      |
| Go (via cgo)            | `IsYes(...)`           | `IsYesP(...)`           |

**Several options instead of yes or no?** `choose` returns the index of the best option and `choose_p` the whole ranking, best first, with probabilities that sum to one:

```python
from decisiongator import choose, choose_p

teams = ["billing", "technical support", "sales"]
message = "My card was charged twice for last month's invoice."

i = choose(message, "Which team should handle this message?", teams, threshold=0.60)
# 0 for "billing"; None when the best option is below 60%, so you can hand it to a person.

ranked = choose_p(message, "Which team should handle this message?", teams)
# [(0, 0.98), (2, 0.01), (1, 0.01)]: (index, probability) pairs, best first.
```

Java is `Decisions.choose(...)` and `Decisions.chooseP(...)`, JavaScript `choose`/`chooseP`, and C `dg_choose`/`dg_choose_p` with caller-owned output arrays and nothing to free. A deferred choice is `-1` outside Python.

**P means probability of yes**, from zero to one. Boolean calls return true when that probability is at least **0.5** by default. Python accepts `threshold=0.90`; Java accepts a threshold overload, as above; C offers `dg_is_yes_at_threshold`. Choose a threshold using examples from your application, or use probabilities to reserve an uncertain range for review. Errors are reported separately, never disguised as “no.”

**Getting started takes three commands and no compiler.** Clone the repository, download the prebuilt binaries for your platform (or all of them), then install the package for your language from `released/`:

```text
git clone https://github.com/CarlFreeAiEngineer/DecisionGator.git
cd DecisionGator
uv run code/fetch_released.py --only python --only macos-arm64   # or linux-x64, windows-x64, java, node, web; omit --only for everything
```

To try it first, [examples/](examples/README.md) has a small program for every language that runs with one command after the download (`--for java`, `--for node`, and so on fetch only what that example needs). To use it in your project, install a [Python wheel](released/python/README.md), add the [Java JARs](java/README.md), install a [Node.js package](javascript/README.md), reference the [C# project](csharp/README.md), link the [C library](code/README.md), or call it from [Rust](examples/rust/) or [Go](examples/go/). The prebuilt bundles are too large for GitHub, so fetch them into `released/` with `uv run code/fetch_released.py` (they come from [62-84-178-253.sslip.io/DecisionGator](https://62-84-178-253.sslip.io/DecisionGator/), checksum-verified). Registry publication comes later. For custom bundles and explicit resource management, see [the interface specification](specs/component-api.md).

## How it compares

Every row below can answer "is this person asking for an appointment?" The differences are the painful parts: what you have to set up, what shape the answer comes back in, and what you do when it's wrong.

| | Setup | The answer comes back as | Offline | Wrong answer? |
| --- | --- | --- | --- | --- |
| **DecisionGator** | Add a library | A boolean | Yes | Add data, retrain |
| **Jev** (TypeSafe AI) | API key, waitlist, network | Typed answer + probability | No | Can't; it's hosted |
| **Laya** (Convai Innovations) | Python package, PyTorch | Typed answer + probability | Yes | Fine-tune in a notebook |
| **Regex / keywords** | None | A boolean, for cases you thought of | Yes | Add a pattern, break another |
| **LLM API** | API key, billing, network | JSON, if you beg | No | Prompt harder and hope |
| **Local LLM** (Ollama, llama.cpp) | Runtime, model files, GPU | JSON, if you beg | Yes | Prompt harder and hope |
| **Needle 3** (Cactus) | Python package or C library | JSON tool call | Yes* | LoRA fine-tune |
| **Zero-shot NLI model** | PyTorch or ONNX, in Python | Logits you threshold | Yes | Write the training code |

\* Needle's engine binary sends telemetry unless you turn it off.

Numbers are from each project's own published material as of September 2026 and will drift; check the source before relying on them.

What the table hides:

- **Jev** answers the way DecisionGator does: a typed answer with a probability, no begging. But it is a hosted service with a waitlist and a per-call bill, your users' text leaves your machine every time, and you cannot retrain it.
- **Laya** is the free, open version of Jev, and it runs on your own machine. In [our side-by-side test](reports/laya-comparison.md) on a laptop CPU it was less accurate (69% against DecisionGator's 90.6%) and no faster on yes/no questions. It needs Python and PyTorch, and downloads its model from Hugging Face on first use. DecisionGator is a file you drop into your project; any programming language.
- **Regex and keyword matching** works until someone writes "no rush, but could I come in Tuesday?" Every odd case needs another pattern, and each new pattern breaks an old one.
- **A hosted LLM** is the most flexible option. It also means an API key, a bill, a network connection, your users' text on someone else's server, and begging for JSON.
- **A local LLM** (Ollama, llama.cpp) removes the network and the bill, but you still install a runtime, manage model files, want a GPU, and beg for JSON.
- **Needle 3** is built for a different job: tool calling on phones and small devices. You can force a yes/no out of it, but there is no simple yes/no function. Its engine is free only for small companies and sends usage data unless you turn that off.
- **A zero-shot NLI model** such as `facebook/bart-large-mnli` is free, but you write the setup, thresholds, and packaging yourself, in Python. DecisionGator is that work done once, for seven languages.

## What if this gives a wrong answer?

# GREAT!

**That means you can add training data, retrain the model yourself easily right here in this project, and contribute your training data to the project so the whole world can benefit!**

A wrong answer from a closed API is a dead end: you file a ticket and hope. A wrong answer here is an improvement you can make yourself. Write down the text, the question, and the right answer, add them to the training data, and retrain. The training data is plain text files in this repository, and the training recipe runs on an ordinary laptop. Keep the improved model for yourself, or send your examples back in a pull request and the next release gets better for everyone.

**Open weights. Open training data. Open recipes.**

1. **Catch a mistake.** Save the content and question, supply the correct answer, and explain why.
2. **Teach your own copy.** Clone the repository, add your examples, and [retrain and test](specs/accuracy-v2.md) a version you can ship.
3. **Share the improvement.** Submit examples through a normal Git pull request. Reviewed contributions improve the shared training data; tested improvements become new releases. The step-by-step version is in [CONTRIBUTING.md](CONTRIBUTING.md).

You need neither a GPU nor machine-learning expertise to contribute an example. You can also retrain privately without sharing your data. A correction is useful evidence, not a guaranteed fix: evaluation checks whether it helps without breaking earlier decisions.

We intend to publish releases and training assets, likely on Hugging Face alongside this repository. Our training data is open; that does not mean we possess the upstream pretraining corpus. See [contribution and training details](specs/open-training.md).

## What you can use today

The experimental Mac bundle is about **620 MB** and the browser bundle about **580 MB**, and a running process holds about 1.5 GB. Version 0.4.0 moved to a 435M-parameter DeBERTa-v3-large foundation because accuracy was the priority; version 0.4.1 stores those weights as 8-bit and 4-bit integers, which made it smaller and faster at a small cost in accuracy. Quantized arithmetic rounds slightly differently on each processor, so a borderline case can get a different answer on different platforms; confident answers match. Both run entirely inside your application, with no telemetry, remote inference fallback, or per-call bill. Browser assets are delivered with your web app. Training ran on a Google Colab A100 (the M1 Pro could not hold the large model). Project code is Apache-2.0 and the 0.4.0 weights derive from an MIT-licensed DeBERTa-v3-large checkpoint; dependencies carry their own [license notices](notices/README.md), which also record a training-data caveat from the upstream model card.

**Accuracy is measured, not guaranteed:** version 0.4.1 answered **435 of 480 held-out yes/no test cases (90.6%)** written by separate authors across sixteen families, **74 of 80 fresh cases (92.5%)** and **18 of 20 held-out multiple-choice cases**; the uncompressed 0.4.0 model scored 92.3%, 92.5%, and 19 of 20, and 0.3.0 scored 59.2%, 77.5%, and 14 of 20 on the same sets. Intent, negation, quoted instructions, routing, and duplicate detection are at or above 90% per family; arithmetic and exact numeric boundaries are the weak spot (70%). All test data is synthetic and machine-authored, so there is still no independent human-reviewed benchmark. See [measured accuracy, size, and speed](reports/accuracy-v4.1.md) and [the 0.4.0 report](reports/accuracy-v4.md).

The current version handles English text, yes/no questions, and multiple choice over caller-supplied options. Other decision types come later.

Building it? Start with the [developer specifications](specs/README.md).
