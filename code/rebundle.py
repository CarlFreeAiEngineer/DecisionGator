#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Put a newly trained model into a copy of a released native bundle, without recompiling anything.

    uv run code/rebundle.py --target linux-x64 --model models/NEW --output models/NEW-linux-x64

The native library and its runtime are taken unchanged from released/<target>; only model.onnx,
tokenizer.json and the manifest are new. The manifest keeps the released checksums of the unchanged
files, so the bundle loads and build_python.py can pack it. Use code/build.py instead when the
native code itself has changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target', choices=['macos-arm64', 'linux-x64', 'windows-x64'], required=True)
    parser.add_argument('--model', type=Path, required=True, help='folder with model.onnx, tokenizer.json, manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-id', help='model_id to record; default: the one in the model folder')
    args = parser.parse_args()
    released = ROOT / 'released' / args.target
    if args.output.exists():
        raise SystemExit('Output exists; choose a fresh directory')
    manifest = json.loads((args.model / 'manifest.json').read_text(encoding='utf-8'))
    old = json.loads((released / 'manifest.json').read_text(encoding='utf-8'))
    for name in ('model.onnx', 'tokenizer.json'):
        if digest(args.model / name) != manifest['sha256'][name]:
            raise SystemExit(f'Model folder checksum mismatch: {name}')
    shutil.copytree(released, args.output, ignore=shutil.ignore_patterns('model.onnx', 'tokenizer.json', 'manifest.json', 'c_smoke*',
                                                                         'build-checks.json', 'platform-parity.json'))
    for name in ('model.onnx', 'tokenizer.json'):
        shutil.copy2(args.model / name, args.output / name)
    manifest['sha256'] = {name: manifest['sha256'][name] for name in ('model.onnx', 'tokenizer.json')}
    manifest['sha256'].update({name: value for name, value in old['sha256'].items() if name not in manifest['sha256']})
    for name, expected in manifest['sha256'].items():
        if digest(args.output / name) != expected:
            raise SystemExit(f'Checksum mismatch after copying: {name}')
    for key in ('parity', 'build'):
        if key in old:
            manifest[key] = old[key]
    manifest['native_from'] = old.get('model_id')  # the release whose native library and runtime this bundle reuses
    if args.model_id:
        manifest['model_id'] = args.model_id
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
