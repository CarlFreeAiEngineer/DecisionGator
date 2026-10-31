#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Run a cross-built bundle's checks on its own platform; no compiler needed.

code/build.py --target linux-x64 or windows-x64 builds a bundle on a Mac and
leaves build-checks.json pending. Copy the bundle to a machine of that platform
with a checkout of this repository, then run:
  uv run tests/target_check.py --bundle BUNDLE
It checks the bundle's hashes, runs the prebuilt C smoke program, the native ABI
and Python tests, and the frozen cross-platform comparison when the model is the
frozen one, then marks build-checks.json passed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from build import host_target  # noqa: E402


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    platform_id = host_target()[0]
    checks_path = bundle / 'build-checks.json'
    pending = json.loads(checks_path.read_text(encoding='utf-8'))
    if pending.get('platform_id') != platform_id:
        raise SystemExit(f'This bundle is for {pending.get("platform_id")}; this machine is {platform_id}.')
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['sha256'].items():
        if sha256(bundle / name) != digest:
            raise SystemExit(f'Hash mismatch: {name}; the copy is incomplete or changed.')
    checks = [check for check in pending['checks'] if 'on the build Mac' in check] or ['Rust unit tests']
    smoke = bundle / ('c_smoke.exe' if platform_id == 'windows-x64' else 'c_smoke')
    if sha256(smoke) != pending.get('smoke_sha256', {}).get(smoke.name):
        raise SystemExit(f'Hash mismatch: {smoke.name}; the copy is incomplete or changed.')
    subprocess.run([str(smoke)], cwd=bundle, check=True)
    checks.append('C host load/inference/exit (cross-compiled smoke program)')
    subprocess.run([sys.executable, str(ROOT / 'tests/native_check.py'), '--bundle', str(bundle)], check=True)
    checks.append('native ABI and Python tests')
    reference_path = ROOT / 'tests/fixtures/platform-parity.json'
    reference = json.loads(reference_path.read_text(encoding='utf-8')) if reference_path.is_file() else {'sha256': {'model.onnx': None}}
    if all(manifest['sha256'].get(name) == digest for name, digest in reference['sha256'].items()) and manifest.get('temperature', 1.0) == reference.get('temperature'):
        subprocess.run([sys.executable, str(ROOT / 'tests/platform_parity.py'), '--bundle', str(bundle), '--output', str(bundle / 'platform-parity.json')], check=True)
        checks.append('frozen platform parity cases')
    else:
        (bundle / 'platform-parity.json').write_text(json.dumps({'status': 'not applicable', 'reason': 'This model differs from the frozen reference. Freeze a new reference with tests/platform_parity.py before claiming cross-platform parity.'}, indent=2) + '\n', encoding='utf-8')
    checks_path.write_text(json.dumps({'status': 'passed', 'checks': checks, 'platform_id': platform_id, 'checked_on': platform.platform()}, indent=2) + '\n', encoding='utf-8')
    print(f'Checks passed on {platform.platform()}: {bundle}')


if __name__ == '__main__':
    main()
