#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""One rule for an adults-only chat site, checked with two yes/no questions (specs/guard-check.md).

A message is blocked when it has sexual content AND mentions a child or anyone under 18. Sexual talk
between adults passes; plain talk about children passes. Needs a DecisionGator model trained for the
second question (0.4.3 or later); 0.4.1 answers it poorly.

    from guard_check import check
    result = check("the message", known=["Mia is the sender's daughter, aged six."], before="what was said just before")
    if result["block"]: ...

Run this file for a small demonstration. Copy it into your application; it has no other dependency
than the decisiongator package.
"""
import re

SEXUAL_QUESTION = 'Does this message contain sexual content?'
CHILD_QUESTION = 'Does the message mention a child or anyone under 18?'
# Optional criteria for the child question. The trained model was taught the question with and without them.
CHILD_CRITERIA = {
    'yes': 'It mentions or refers to a child, baby, teenager or anyone under 18, also by name or by she, he or they '
           'when Known or Before shows that person is under 18, and the writer\'s own childhood.',
    'no': 'Everyone it mentions is an adult. Pet names for an adult (baby, girl, boy, little man, daddy) do not count, '
          'nor does a child who is only in Known or Before.',
}
SEXUAL_THRESHOLD = 0.03  # at or above: the message has sexual content. Low on purpose: every doubt blocks.
CHILD_THRESHOLD = 0.20   # at or above: a child is mentioned.
PIECE_CHARS = 320        # a long message is read in overlapping pieces of about this many characters
KNOWN_CHARS = 160        # background is cut to this much, to stay inside the model's 256-token input
BEFORE_CHARS = 160


def pieces(text, limit=PIECE_CHARS):
    """Cut a message into pieces of whole sentences, each starting with the last sentence of the one before."""
    text = ' '.join(text.split())
    sentences = []
    for sentence in re.findall(r'.+?(?:[.!?]["\')*]*(?=\s)|$)', text):
        sentence = sentence.strip()
        while len(sentence) > limit:  # a sentence too long for one piece: cut at a space
            cut = sentence.rfind(' ', 0, limit)
            cut = cut if cut > limit // 2 else limit
            sentences.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            sentences.append(sentence)
    result, current = [], []
    for sentence in sentences:
        if current and len(' '.join(current + [sentence])) > limit:
            result.append(' '.join(current))
            current = [current[-1]] if len(current[-1]) + len(sentence) < limit else []
        current.append(sentence)
    if current:
        result.append(' '.join(current))
    return result or [text]


def child_content(piece, known=(), before=''):
    """What the child question reads: the background lines, when there are any, then the piece."""
    known = ' '.join(k.strip() for k in ([known] if isinstance(known, str) else known) if k and k.strip())[:KNOWN_CHARS]
    before = ' '.join((before or '').split())[-BEFORE_CHARS:]
    lines = (['Known: ' + known] if known else []) + (['Before: ' + before] if before else [])
    return '\n'.join(lines + ['Message: ' + piece]) if lines else piece


def check(text, known=(), before='', *, score=None, criteria=None, sexual_threshold=SEXUAL_THRESHOLD,
          child_threshold=CHILD_THRESHOLD, always_both=False):
    """Judge one message. `known`: short sentences about the children in the people's lives; `before`: the message
    sent just before (never judged itself). Returns block, sexual and child (the highest probability over the
    pieces) and the number of model calls. A message with no sexual content passes without the child question.
    `score(content, question, criteria)` defaults to decisiongator.is_yes_p."""
    if score is None:
        from decisiongator import is_yes_p as score
    parts = pieces(text)
    calls = len(parts)
    sexual = max(score(part, SEXUAL_QUESTION, None) for part in parts)
    child = None
    if sexual >= sexual_threshold or always_both:
        child = max(score(child_content(part, known, before), CHILD_QUESTION, criteria) for part in parts)
        calls += len(parts)
    return {'block': sexual >= sexual_threshold and child is not None and child >= child_threshold,
            'sexual': sexual, 'child': child, 'calls': calls}


if __name__ == '__main__':
    for message, known in [
        ('My daughter starts school on Monday and I am so proud of her.', []),
        ('Come to bed. I have wanted you all day.', []),
        ('She is asleep upstairs at last. Come to bed, I have wanted you all day.', ["Mia is the sender's daughter, aged six."]),
    ]:
        print(check(message, known), '|', message)
