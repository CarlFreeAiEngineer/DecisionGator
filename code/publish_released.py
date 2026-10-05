#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Publish the local released/ tree to https://62-84-178-253.sslip.io/DecisionGator/files/<version>/.

Maintainers only: needs SSH access to the web host. Writes SHA256SUMS.txt from
the local files, then copies everything with rsync into a versioned folder.
code/fetch_released.py reads that same SHA256SUMS.txt, so publishing a version
is what makes it downloadable. --site also deploys the pages in website/;
--site-only deploys just those.

  uv run code/publish_released.py --version 0.4.1
  uv run code/publish_released.py --version 0.4.1 --site
  uv run code/publish_released.py --site-only
  uv run code/publish_released.py --version 0.4.1 --dry-run
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = 'ace@62.84.178.253'
DEFAULT_REMOTE_DIR = '/var/www/textautomationlib/DecisionGator/files'
SITE_PAGES = {'index.html': 'index.html', 'try.html': 'try.html', 'files-index.html': 'files/index.html'}  # website/ name -> path under DecisionGator/
SKIP_DIRS = {'__pycache__', 'node_modules'}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_sums(source):
    lines = [f'# DecisionGator release files. Verify with: sha256sum -c SHA256SUMS.txt']
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.name == 'SHA256SUMS.txt' or path.suffix == '.part':
            continue
        if SKIP_DIRS & set(path.relative_to(source).parts):
            continue
        lines.append(f'{sha256(path)}  {path.relative_to(source).as_posix()}')
    (source / 'SHA256SUMS.txt').write_text('\n'.join(lines) + '\n')
    return len(lines) - 1


def publish_site(host, remote_dir, dry_run):
    site_root = remote_dir.rsplit('/files', 1)[0]
    for local, remote in SITE_PAGES.items():
        source = ROOT / 'website' / local
        if not source.is_file():
            sys.exit(f'{source} is missing')
        target = f'{host}:{site_root}/{remote}'
        print(f'{source.relative_to(ROOT)} -> {target}')
        if not dry_run:
            subprocess.run(['rsync', '-z', '-p', '--chmod=ugo=rwX,go-w', str(source), target], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version', help='folder name on the server, for example 0.4.1')
    parser.add_argument('--site', action='store_true', help='also deploy the pages in website/')
    parser.add_argument('--site-only', action='store_true', help='deploy only the pages in website/')
    parser.add_argument('--source', type=Path, default=ROOT / 'released')
    parser.add_argument('--host', default=DEFAULT_HOST)
    parser.add_argument('--remote-dir', default=DEFAULT_REMOTE_DIR, help='directory on the web host (relative paths are under the SSH user home)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if not shutil.which('rsync'):
        sys.exit('rsync is required; install it with your package manager')
    if args.site_only:
        publish_site(args.host, args.remote_dir, args.dry_run)
        return
    if not args.version:
        parser.error('--version is required unless --site-only is given')
    if not args.source.is_dir():
        sys.exit(f'{args.source} does not exist')
    count = write_sums(args.source)
    print(f'{count} files listed in {args.source / "SHA256SUMS.txt"}')

    destination = f'{args.host}:{args.remote_dir}/{args.version}/'
    subprocess.run(['ssh', args.host, f'mkdir -p {args.remote_dir}/{args.version}'], check=True)
    command = ['rsync', '-az', '--partial', '--chmod=ugo=rwX,go-w']  # world-readable so the web server can serve; symbolic form works in both rsync and openrsync
    command += [f'--exclude={d}' for d in SKIP_DIRS] + ['--exclude=*.part']
    if args.dry_run:
        command.append('--dry-run')
    command += [f'{args.source}/', destination]
    subprocess.run(command, check=True)
    if not args.dry_run:
        # --chmod does not fix files that were already on the server unchanged; unreadable ones give users a 403
        subprocess.run(['ssh', args.host, f'chmod -R a+rX {args.remote_dir}/{args.version}'], check=True)
    print(f'published {args.version} to https://62-84-178-253.sslip.io/DecisionGator/files/{args.version}/')
    if args.site:
        publish_site(args.host, args.remote_dir, args.dry_run)


if __name__ == '__main__':
    main()
