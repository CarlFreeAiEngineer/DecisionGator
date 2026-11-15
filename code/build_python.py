#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Package an existing native bundle as a dependency-free platform wheel."""
import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = '0.4.1'
# The bundled Mac runtime requires 13.3. A 14.0 tag conservatively avoids
# advertising compatibility with 13.0-13.2 (macOS wheel tags use major versions).
TARGETS = {
    'macos-arm64': ('macosx_14_0_arm64', 'libdecisiongator.dylib'),
    'linux-x64': ('linux_x86_64', 'libdecisiongator.so'),
    'windows-x64': ('win_amd64', 'decisiongator.dll'),
}


def main():
    global VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=TARGETS, required=True)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'released/python')
    parser.add_argument('--version', default=VERSION, help='package version, when the bundle holds a newer model')
    args = parser.parse_args()
    VERSION = args.version
    platform_tag, library = TARGETS[args.target]
    bundle = (args.bundle or ROOT / 'released' / args.target).resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    if not (bundle / library).is_file():
        raise SystemExit(f'Missing {library}')
    for name, expected in manifest['sha256'].items():
        path = bundle / name
        if not path.resolve().is_relative_to(bundle):
            raise SystemExit(f'Invalid manifest path: {name}')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise SystemExit(f'Checksum mismatch: {name}')
    args.output.mkdir(parents=True, exist_ok=True)
    tag = f'py3-none-{platform_tag}'
    destination = args.output / f'decisiongator-{VERSION}-{tag}.whl'
    info = f'decisiongator-{VERSION}.dist-info'
    records = []
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as wheel:
        def add(name, data):
            wheel.writestr(name, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
            records.append((name, 'sha256=' + digest, len(data)))

        add('decisiongator/__init__.py', (ROOT / 'decisiongator/__init__.py').read_bytes())
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                add('decisiongator/_bundle/' + path.relative_to(bundle).as_posix(), path.read_bytes())
        add(info + '/METADATA', (
            f'Metadata-Version: 2.1\nName: decisiongator\nVersion: {VERSION}\n'
            'Summary: Offline yes/no decisions as an ordinary software component\n'
            'Requires-Python: >=3.11\nLicense: Apache-2.0\n'
            'Description-Content-Type: text/markdown\n\n'
            'Includes the native component and all inference assets. No runtime downloads.\n'
            'Bundled weights and dependencies retain their own licenses under decisiongator/_bundle/notices/.\n'
        ).encode())
        add(info + '/WHEEL', (
            'Wheel-Version: 1.0\nGenerator: decisiongator-build-python\n'
            f'Root-Is-Purelib: false\nTag: {tag}\n'
        ).encode())
        add(info + '/LICENSE', (ROOT / 'LICENSE').read_bytes())
        record = io.StringIO(newline='')
        csv.writer(record, lineterminator='\n').writerows([*records, (info + '/RECORD', '', '')])
        wheel.writestr(info + '/RECORD', record.getvalue())
    print(destination)


if __name__ == '__main__':
    main()
