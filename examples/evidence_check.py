#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["onnxruntime>=1.22", "tokenizers>=0.20", "numpy"]
# ///
"""Evidence check 0.2 (experimental): which sentences of a reply add information the evidence does not hold?

Give it the text a reply was written from (the evidence: a chat history, notes, a source document)
and the reply. It judges every sentence of the reply in one pass and prints, per sentence, the
probability that it is supported, unsupported (clearly new information), or not a claim at all (a
question, a feeling, a plan). Read released/evidence-check-0.2/README.md before relying on it: it
catches about 7 in 10 clear inventions and flags about 1 in 23 honest messages. It does not look for
known facts told wrongly. See reports/evidence-check-v1.md.

  uv run examples/evidence_check.py --evidence notes.txt --reply reply.txt
  uv run examples/evidence_check.py --demo

The first run downloads the model (about 600 MB) into released/evidence-check-0.2/ and checks its hash.
"""
import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'evidence-check-0.2'
MODEL_DIR = ROOT / 'released' / VERSION
BASE_URL = f'https://62-84-178-253.sslip.io/DecisionGator/files/{VERSION}'
LABELS = ['supported', 'unsupported', 'not_claim']

DEMO_EVIDENCE = """Maya: I finally adopted a dog last weekend! She's a three-year-old beagle named Pepper.
Sam: That's wonderful news. How is she settling in?
Maya: Slowly. She hid under the kitchen table for the first two days, but last night she slept on the rug by my bed.
Maya: The shelter said she was found near the train station in March.
Sam: Poor thing. Does she eat well?
Maya: Yes, she loves the chicken kibble, and my neighbor Rosa gave us a bag of treats.
Maya: I took her to the vet on Thursday and she weighs eleven kilos, which the vet said is healthy.
Sam: Any plans for the week?
Maya: I'm nervous about leaving her alone when I go back to the office on Monday."""
DEMO_REPLY = ("Congratulations on adopting Pepper! A three-year-old beagle is a great age. "
              "It's sweet that the shelter found her near the train station. "
              "I remember you said she loves swimming in the lake. "
              "How are you feeling about Monday?")


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model():
    """Download any missing file, verified against the published SHA256SUMS.txt."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f'{BASE_URL}/SHA256SUMS.txt', timeout=60) as response:
        sums = dict(reversed(line.split(None, 1)) for line in response.read().decode().splitlines()
                    if line.strip() and not line.startswith('#'))
    for name in ('manifest.json', 'tokenizer.json', 'model.onnx'):
        path = MODEL_DIR / name
        if path.is_file() and sha256(path) == sums[name]:
            continue
        print(f'downloading {name} ...', file=sys.stderr)
        part = path.with_suffix(path.suffix + '.part')
        urllib.request.urlretrieve(f'{BASE_URL}/{name}', part)
        if sha256(part) != sums[name]:
            part.unlink()
            sys.exit(f'{name}: checksum mismatch, download again')
        part.replace(path)


def split_sentences(text):
    """A plain splitter: after . ! ? followed by a space. Good enough for chat replies."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


class EvidenceCheck:
    def __init__(self, model_dir=MODEL_DIR, threads=None):
        self.manifest = json.loads((model_dir / 'manifest.json').read_text())
        self.tokenizer = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_dir / 'model.onnx'), options, providers=['CPUExecutionProvider'])
        self.cls = self.tokenizer.token_to_id('[CLS]')
        self.sep = self.tokenizer.token_to_id('[SEP]')

    def check(self, evidence, sentences, max_tokens=4096):
        """Per sentence, a dict of probabilities for supported / unsupported / not_claim.
        Only the evidence is cut to fit max_tokens, from the front (the newest part is kept).
        The model was trained at 4,096 tokens. 2,048 keeps a CPU call to about 2.5 seconds, but
        cutting evidence the reply relied on more than doubles false alarms."""
        text, starts = 'Reply:', []
        for i, s in enumerate(sentences, 1):
            text += ' '
            starts.append(len(text))
            text += f'[{i}] {s}'
        if len(evidence.split()) < 80:
            print('warning: under about 100 tokens of evidence the model flags nearly everything; '
                  'it was trained on evidence of 400 tokens and more', file=sys.stderr)
        reply = self.tokenizer.encode(text, add_special_tokens=False)
        a = reply.ids
        if len(a) > max_tokens // 2:
            sys.exit('the reply is too long for one call; split it')
        positions = [next(t for t, (_, end) in enumerate(reply.offsets) if end > c) + 1 for c in starts]  # +1 for [CLS]
        b = self.tokenizer.encode('Evidence:\n' + evidence, add_special_tokens=False).ids
        room = max_tokens - 3 - len(a)
        b = b[max(0, len(b) - room):]
        ids = [self.cls] + a + [self.sep] + b + [self.sep]
        logits = self.session.run(None, {
            'input_ids': np.array([ids], dtype=np.int64),
            'attention_mask': np.ones((1, len(ids)), dtype=np.int64),
            'marker_positions': np.array([positions], dtype=np.int64),
        })[0][0] / self.manifest['temperature']
        p = np.exp(logits - logits.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        return [dict(zip(LABELS, map(float, row))) for row in p]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--evidence', type=Path, help='text file the reply was written from')
    parser.add_argument('--reply', type=Path, help='text file with the reply to check')
    parser.add_argument('--demo', action='store_true', help='run a built-in example')
    parser.add_argument('--threshold', type=float, default=0.5, help='flag a sentence when p(unsupported) is at least this')
    parser.add_argument('--max-tokens', type=int, default=4096, help='evidence beyond this is cut from the front')
    args = parser.parse_args()
    if args.demo:
        evidence, reply = DEMO_EVIDENCE, DEMO_REPLY
    elif args.evidence and args.reply:
        evidence, reply = args.evidence.read_text(), args.reply.read_text()
    else:
        parser.error('give --evidence and --reply, or --demo')
    ensure_model()
    sentences = split_sentences(reply)
    for sentence, p in zip(sentences, EvidenceCheck().check(evidence, sentences, args.max_tokens)):
        flag = 'FLAG' if p['unsupported'] >= args.threshold else '    '
        print(f"{flag} unsupported {p['unsupported']:.2f}  supported {p['supported']:.2f}  not_claim {p['not_claim']:.2f}  {sentence}")


if __name__ == '__main__':
    main()
