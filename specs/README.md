# DecisionGator developer specifications

Status: experimental component v0.4, 2026-09-19 (Mac and browser); Windows and Linux bundles remain v0.2 until rebuilt on those platforms. Native builds, Python, Java, Node.js/TypeScript, browser WebAssembly, open training data, and the training pipeline are implemented. Platform qualification is recorded in the [release inventory](../released/README.md) and test reports. Version 0.4 reached 92% on a 480-case held-out synthetic test; the decision-quality requirements below still lack a human-reviewed benchmark. Start with [the v0.4 results](../reports/accuracy-v4.md), then [the earlier recipe](accuracy-v2.md), [the v0.3 results](../reports/accuracy-v3.md), and [the v0.2 results](../reports/accuracy-v2.md). The [first-version recipe](first-version.md) remains available for reproducing the original experiment.

## Objective

Build a reusable offline component that estimates the probability of yes for a question supplied at runtime, given caller-supplied text and optional decision criteria, and that ranks a caller-supplied list of options for a question. Ship the component and its trained weights with the application.

The product goal is general reuse across question types, not a fixed refund detector or a separate trained model per question. Generalization to unseen questions is therefore a release criterion, not a future enhancement.

## Read in order

1. [Behavior and interface](behavior.md): inputs, probability meaning, validation, and caller control.
2. [Model and training](model-and-training.md): candidate architectures, data, calibration, and local experiments.
3. [Runtime and distribution](runtime.md): native packaging, offline operation, portability, and licensing.
4. [Evaluation and milestones](evaluation.md): evidence required to proceed and release.
5. [Open training and contributions](open-training.md): public data, reproducible releases, and community improvements.
6. [Research notes](research.md): transcript findings, primary sources, and unresolved assumptions.
7. [Development and builds](development.md): Rust, portable tools, native builds, and the three release targets.
8. [Ordinary component interfaces](component-api.md): automatic initialization, simple calls, and Java packaging.
9. [Node.js and WebAssembly](javascript-and-webassembly.md): implemented bindings, shared weights, browser deployment, and verification.
10. [Evidence check](evidence-check.md): a second, long-input model that says whether a long text supports a sentence (in progress).

## Firm requirements

- All inference runs locally with no runtime network dependency, usage metering, or mandatory accelerator.
- Integrate as an ordinary in-process library: no Ollama, llama.cpp, agent harness, separately managed model runner, or server process.
- The question is an input. Changing the question must not require retraining.
- Return a finite probability estimate, never generated prose.
- Keep application actions and decision thresholds outside the learned model.
- Support bundling and royalty-free redistribution through compatible code, model, tokenizer, and data permissions.
- Publish model weights, project training data, and reproducible training recipes; accept community examples, corrections, and model improvements.
- Make clone, add data, validate, retrain, compare, and export a supported workflow. Allow private local examples and ordinary Git pull requests for shared improvements; contributing data must not require training hardware.
- Publish measured limitations. Do not equate valid output types with correct judgments.
- Keep project knowledge, experiment recipes, and release evidence in this repository.
- Implement the native component in Rust using a project-local toolchain under `tools/`; build each release target on its own platform.

## Current scope

English text, one content and one question per call, optional yes/no criteria, multiple choice over two to 256 caller-supplied options, desktop CPU inference, and reproducible local training. Native targets are macOS arm64, Windows x86-64, and Linux x86-64; each platform is qualified by its own build and tests. Node.js and browser WebAssembly are implemented as described in the [JavaScript specification](javascript-and-webassembly.md), using the same trained artifact. Other architectures, mobile, images, ratings, free-text extraction, and shared-content multi-question acceleration are deferred.

The current artifact uses a 256-token limit; its size, licenses, accuracy, and measured performance are recorded in the release docs and reports. Those measurements describe the experimental release, not guarantees for every device or future version.
