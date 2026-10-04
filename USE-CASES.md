# Less obvious uses for DecisionGator

The obvious uses are classifying customer messages: is this a refund request, which team handles it, is it urgent. This page collects the less obvious ones: places where a fast, private, offline yes/no or pick-one decision quietly makes other software better. Add your own.

Keep in mind what DecisionGator is good and bad at when you pick a use:

- It reads up to 256 tokens (roughly 200 words) of content per call; longer text is cut off.
- A yes/no call takes roughly 100 to 300 milliseconds on a laptop CPU. A multiple-choice call runs once per option, so 10 options cost about 10 calls.
- It is strong on intent, meaning, negation, and "is this the same thing?" questions, and weak on arithmetic and exact number boundaries.
- English only, for now.
- Probabilities let you keep an "unsure, send it to a person" range. Use them.

## Routing a prompt to the right LLM

Send each user prompt to the model best suited for it, before spending any money on a large one.

```python
models = ["small fast model: chit-chat, simple facts, rewording",
          "code model: writing, fixing or explaining code",
          "large reasoning model: multi-step problems, planning, analysis",
          "vision model: the user refers to an image or screenshot"]
best = choose_p(prompt, "Which model should answer this request?", models)[0]
```

Most traffic in a chat product is simple, so sending it to a cheap model saves real money, and the routing decision costs nothing per call and never leaves the machine. Below a confidence cutoff, send it to the large model rather than risk a poor answer. Prompts longer than about 200 words are judged on their beginning only.

## Does this LLM answer actually answer the question?

Check a generated answer before showing it: `is_yes(answer, f'Does this text answer the question "{question}"?')`. A "no" can trigger a retry, a different model, or a "let me get a person" reply. It also catches refusals ("I can't help with that") in a structured way instead of string matching.

## Guarding tool calls in an AI agent

Before an agent runs a command or sends an email, ask: "Does this action delete or overwrite data?", "Does this send something to someone outside the company?" A yes can require human approval. The check runs locally, so it adds no API cost and works even when the agent's own model is the one that went wrong.

## Picking which retrieved passages to give an LLM

In retrieval-augmented generation, ask of each retrieved chunk: `is_yes_p(chunk, 'Does this passage help answer "..."?')` and keep only the best. Our [reranking report](reports/rag-rerank/README.md) found the ranking competitive but the probabilities too high for on-topic passages that do not actually answer, so rank by probability rather than trusting a fixed cutoff.

## Spotting prompt injection in user content

Before untrusted text (an email, a web page, a document) reaches an LLM, ask: "Does this text try to give instructions to an AI assistant?" Quoted instructions are one of the families DecisionGator is tested on. It is one layer of defense, not a guarantee.

## Deciding whether a message needs a reply at all

For notification systems, inboxes and chat bots: "Is the sender asking a question or requesting something?" A "no" ("thanks!", "ok, sounds good") can be closed automatically or left unanswered instead of triggering an auto-response.

## Duplicate and "already reported" detection

Compare a new bug report, support ticket or forum post against recent ones: `is_yes(new_ticket, f'Does this describe the same problem as: "{old_ticket}"?')`. Run it only against a short list found by a cheap keyword or embedding search first, since every comparison is one call.

## Choosing which help article or form to show

Instead of a search box that matches words, pick from your list of help pages or forms by meaning: `choose(user_text, "Which page would help this person?", page_titles)`. With a cutoff, fall back to the search box when it is unsure.

## Smarter logs and alerts

Run on error messages or log lines that already passed a cheap filter: "Does this error mean customers cannot use the service?", "Is this a security problem?" A yes pages someone; a no goes in the daily summary. Everything stays on your own servers, which matters for logs full of private data.

## Keeping private data out of places it should not go

Before text is pasted into a shared channel, sent to an outside API, or written to a log: "Does this text contain a password, API key or personal medical information?" Pattern matching finds key formats; DecisionGator catches the ones described in words ("my password is my dog's name plus 1990").

## Form and data-entry sanity checks

On free-text fields: "Is this a real street address?", "Does this describe a job title?", "Is this a complaint rather than a compliment?" in a feedback form with a star rating. A mismatch between the stars and the text is worth a second look.

## Accessibility and safety in games and chat

In a game or community chat: "Is this player asking for help?", "Is this message bullying someone?" It runs on the player's own device, including in the browser, so nothing is sent to a server just to be checked.

## Ideas to try

Uses nobody has measured yet. If you try one, tell us how it went.

- Deciding which of several translations or summaries best keeps the original meaning.
- Grading short answers in a quiz against the expected answer ("Does this answer say the same thing as …?").
- Sorting a personal photo library by its captions or file notes.
- Choosing which of a smart home's commands a spoken request meant, entirely offline.
