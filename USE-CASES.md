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

## Spotting prompt injection in user content

Before untrusted text (an email, a web page, a document) reaches an LLM, ask: "Does this text try to give instructions to an AI assistant?" Quoted instructions are one of the question families DecisionGator is tested on. It is one layer of defense, not a guarantee.

## Spotting spam

Comments, contact forms, reviews, chat: `is_yes(comment, "Is this an unsolicited sales pitch or spam?")`. Version 0.4.1 got 30 of 32 right on the project's held-out spam test, which includes hard cases such as people complaining about spam or quoting it. Criteria let each site set its own policy, for example whether "message me privately" counts as spam. Spam written to look like normal text, or hidden in links and odd spellings, has not been measured.

## Choosing the right help article

Instead of a search box that matches words, pick from your list of help pages by meaning: `choose(user_text, "Which page would help this person?", page_titles)`. With a cutoff, fall back to the search box when it is unsure. A short description after each title helps: in a quick test routing messages to departments, descriptions made the model more confident on right answers without adding wrong ones. With many pages, narrow to the top ten with a cheap keyword search first, since each option is one call.

## Recognizing the language of a message

Useful for sending a message to a translator or to staff who speak that language. This works only in its simplest form: "Is this text written in English?" got 14 of 16 short messages right in a quick test (Russian came out at 0.51, and English with a French greeting was called not English). Asking it to name the language from a list does not work: it called almost everything English and got 4 of 16 right, because the model was trained on English only. To name the language, use a dedicated language-detection library and use DecisionGator for the decisions after that.

## Recognizing private information in a message

Before text is pasted into a shared channel, sent to an outside API, or written to a log: "Does this text contain a password, API key, or personal medical or financial information?" Pattern matching finds known formats such as card numbers; DecisionGator catches ones described in words ("my password is my dog's name plus 1990"). Use both, and treat a "no" as "probably fine", not as a guarantee.
