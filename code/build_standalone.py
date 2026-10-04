#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Build the one-file DecisionGator library from an existing native bundle.

The result is the same library as the bundle's, with manifest.json, tokenizer.json,
and model.onnx compiled in and ONNX Runtime linked statically, so an application
ships one file plus the license notices. It is written to
released/standalone-<platform>/ (override with --output).

  uv run code/build_standalone.py                      # uses released/<this platform>
  uv run code/build_standalone.py --bundle released/windows-x64

Needs the project Rust toolchain (uv run code/setup.py) and network access on the
first build: the ort crate downloads Microsoft's ONNX Runtime as a static library.
On Windows that library needs MSVC 14.44 (Visual Studio 2022 17.14) or newer.
Only the Windows build has been run so far.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / 'code' / 'standalone'


def host_platform():
    system, machine = platform.system(), platform.machine().lower()
    if system == 'Windows' and machine in ('amd64', 'x86_64'):
        return 'windows-x64', 'decisiongator_standalone.dll'
    if system == 'Darwin' and machine == 'arm64':
        return 'macos-arm64', 'libdecisiongator_standalone.dylib'
    if system == 'Linux' and machine in ('x86_64', 'amd64'):
        return 'linux-x64', 'libdecisiongator_standalone.so'
    sys.exit(f'unsupported host: {system} {machine}')


def main():
    name, library = host_platform()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'released' / name,
                        help='native bundle with manifest.json, model.onnx, tokenizer.json, notices/')
    parser.add_argument('--output', type=Path, default=ROOT / 'released' / f'standalone-{name}')
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    for required in ('manifest.json', 'model.onnx', 'tokenizer.json', 'decisiongator.h', 'notices'):
        if not (bundle / required).exists():
            sys.exit(f'{bundle / required} is missing; fetch it with: uv run code/fetch_released.py --only {name}')
    tools = ROOT / 'tools'
    cargo = tools / 'rust' / 'bin' / ('cargo.exe' if os.name == 'nt' else 'cargo')
    if not cargo.exists():
        sys.exit(f'{cargo} is missing; run: uv run code/setup.py')

    env = dict(os.environ, CARGO_HOME=str(tools / 'cargo'), DG_EMBED_DIR=str(bundle))
    env['PATH'] = str(cargo.parent) + os.pathsep + env.get('PATH', '')
    subprocess.run([str(cargo), 'build', '--release', '--locked'], cwd=CRATE, env=env, check=True)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CRATE / 'target' / 'release' / library, output / library)
    shutil.copy2(bundle / 'decisiongator.h', output / 'decisiongator.h')
    shutil.copytree(bundle / 'notices', output / 'notices', dirs_exist_ok=True)
    shutil.copy2(CRATE / 'README.md', output / 'README.md')
    print(f'{output / library}: {(output / library).stat().st_size / 1e6:.0f} MB')


if __name__ == '__main__':
    main()
