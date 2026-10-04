#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Download the prebuilt DecisionGator release bundles into released/.

The bundles (weights, native libraries, wheels, JARs, npm tarballs) are too
large for GitHub, so they live at https://62-84-178-253.sslip.io/DecisionGator/files/.
This script fetches a version's SHA256SUMS.txt, downloads every listed file
(or only the parts you ask for), verifies each hash, and skips files that are
already present and correct.

Examples:
  uv run code/fetch_released.py                       # everything, latest known version
  uv run code/fetch_released.py --only macos-arm64    # one native bundle
  uv run code/fetch_released.py --only python --only web
  uv run code/fetch_released.py --for java            # what examples/java needs on this computer
  uv run code/fetch_released.py --version 0.4.1 --list
"""
import argparse
import hashlib
import platform
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = 'https://62-84-178-253.sslip.io/DecisionGator/files'
DEFAULT_VERSION = '0.4.1'
CHUNK = 1 << 20


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def read_sums(version):
    url = f'{BASE_URL}/{version}/SHA256SUMS.txt'
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            text = response.read().decode()
    except urllib.error.HTTPError as error:
        sys.exit(f'{url}: HTTP {error.code}. Is {version} published?')
    entries = []
    for line in text.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        digest, name = line.split(None, 1)
        entries.append((digest, name.lstrip('*')))
    if not entries:
        sys.exit(f'{url} lists no files')
    return entries


LANGUAGES = ('python', 'c', 'rust', 'go', 'csharp', 'java', 'node', 'browser')


def needed_for(language, version):
    """Path prefixes one example in examples/ needs on this computer."""
    system, machine = platform.system(), platform.machine().lower()
    native = {('Darwin', 'arm64'): 'macos-arm64', ('Linux', 'x86_64'): 'linux-x64',
              ('Windows', 'amd64'): 'windows-x64', ('Windows', 'x86_64'): 'windows-x64'}.get((system, machine))
    if language == 'browser':
        return ['web/']
    if native is None:
        sys.exit(f'no prebuilt bundle for {system} {machine}; the browser example still works')
    if language == 'java':
        return [f'java/decisiongator-java-{version}.jar', f'java/decisiongator-java-{version}-{native}.jar']
    if language == 'node':
        node = native.replace('macos', 'darwin').replace('windows', 'win32')
        return [f'node/decisiongator-{version}-{node}.tgz']
    return [f'{native}/']


def download(url, target, expected, size_hint=''):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    digest = hashlib.sha256()
    done = 0
    request = urllib.request.Request(url, headers={'User-Agent': 'decisiongator-fetch'})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as out:
        total = int(response.headers.get('Content-Length') or 0)
        while chunk := response.read(CHUNK):
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if total:
                print(f'\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB', end='', flush=True)
    print()
    if digest.hexdigest() != expected:
        partial.unlink(missing_ok=True)
        sys.exit(f'{url}: checksum mismatch; download aborted')
    partial.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version', default=DEFAULT_VERSION, help=f'release version to fetch (default {DEFAULT_VERSION})')
    parser.add_argument('--only', action='append', default=[], metavar='DIR',
                        help='top-level bundle directory to fetch: macos-arm64, linux-x64, windows-x64, standalone-windows-x64 (one-file library), web, python, java, node; repeatable')
    parser.add_argument('--for', dest='language', choices=LANGUAGES,
                        help='fetch only what that language\'s example in examples/ needs on this computer')
    parser.add_argument('--output', type=Path, default=ROOT / 'released', help='destination directory (default released/)')
    parser.add_argument('--list', action='store_true', help='list the published files and exit')
    args = parser.parse_args()

    entries = read_sums(args.version)
    if args.language:
        prefixes = needed_for(args.language, args.version)
        entries = [e for e in entries if any(e[1].startswith(prefix) for prefix in prefixes)]
    if args.only:
        entries = [e for e in entries if e[1].split('/', 1)[0] in args.only]
        if not entries:
            sys.exit(f'nothing published under {args.only} for {args.version}')
    if args.list:
        for _, name in entries:
            print(name)
        return

    fetched = skipped = 0
    for expected, name in entries:
        target = args.output / name
        if target.is_file() and sha256(target) == expected:
            skipped += 1
            continue
        print(f'{name}')
        download(f'{BASE_URL}/{args.version}/{name}', target, expected)
        fetched += 1
    print(f'{fetched} downloaded, {skipped} already present, into {args.output}')


if __name__ == '__main__':
    main()
