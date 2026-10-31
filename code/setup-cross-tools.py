#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Add the tools a Mac needs to build the Linux x64 and Windows x64 bundles too.

Run: uv run code/setup-cross-tools.py

Needs the project-local Rust in tools/rust (uv run code/setup.py) and Homebrew
LLVM (brew install llvm@22), whose clang-cl and llvm-lib compile the C parts of
the tokenizer for Windows. Adds, only where missing:

- the Rust standard library for each foreign target, matching tools/rust
- Zig in tools/zig and cargo-zigbuild in tools/bin: C compiler and linker for Linux
- cargo-xwin in tools/bin: builds Visual Studio-style Windows DLLs; it downloads
  Microsoft's C runtime and Windows SDK libraries into tools/xwin on first use
  (by using it you accept Microsoft's license terms for those files)
- ONNX Runtime for Linux x64 and Windows x64 in tools/onnxruntime/<platform>,
  taken from the official Python wheels and checked against PyPI's hashes

Safe to run again. After a Rust upgrade, delete the two foreign folders under
tools/rust/lib/rustlib and rerun to fetch matching standard libraries.
"""
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'

CROSS_TARGETS = ['x86_64-unknown-linux-gnu', 'x86_64-pc-windows-msvc']
# Pinned so every machine builds with the same tools; bump deliberately.
ZIG_VERSION = '0.16.0'
ZIGBUILD_VERSION = '0.23.4'
XWIN_VERSION = '0.23.1'
ORT_VERSION = '1.22.1'
# Wheel filename endings for each platform's ONNX Runtime (the CPython 3.12 wheels, as native builds used before).
ORT_WHEELS = {'linux-x64': 'manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl', 'windows-x64': 'win_amd64.whl'}
LLVM_TOOLS = ('clang-cl', 'llvm-lib')


def download(url, dest):
    print(f'downloading {url}')
    with urllib.request.urlopen(url, timeout=300) as response, open(dest, 'wb') as out:
        shutil.copyfileobj(response, out)


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rustc_version():
    rustc = TOOLS / 'rust/bin/rustc'
    if not rustc.is_file():
        sys.exit(f'No Rust toolchain at {rustc}. Run uv run code/setup.py first.')
    for line in subprocess.check_output([str(rustc), '-vV'], text=True).splitlines():
        if line.startswith('release:'):
            return line.split(':', 1)[1].strip()
    sys.exit('Could not read the rustc version.')


def install_rust_std(version, target, tmp):
    rustlib = TOOLS / 'rust/lib/rustlib'
    if (rustlib / target).is_dir():
        print(f'rust-std {target}: present')
        return
    name = f'rust-std-{version}-{target}'
    archive = tmp / f'{name}.tar.xz'
    download(f'https://static.rust-lang.org/dist/{name}.tar.xz', archive)
    with tarfile.open(archive) as tf:
        tf.extractall(tmp, filter='data')
    shutil.copytree(tmp / name / f'rust-std-{target}/lib/rustlib' / target, rustlib / target)
    print(f'rust-std {target}: installed')


def install_zig(tmp):
    dest = TOOLS / 'zig'
    if (dest / 'zig').is_file():
        print('zig: present')
        return
    arch = {'arm64': 'aarch64', 'x86_64': 'x86_64'}[platform.machine()]
    with urllib.request.urlopen('https://ziglang.org/download/index.json') as response:
        entry = json.load(response)[ZIG_VERSION][f'{arch}-macos']
    archive = tmp / 'zig.tar.xz'
    download(entry['tarball'], archive)
    if sha256(archive) != entry['shasum']:
        sys.exit('zig download does not match its published checksum.')
    with tarfile.open(archive) as tf:
        tf.extractall(tmp, filter='data')
    shutil.move(tmp / f'zig-{arch}-macos-{ZIG_VERSION}', dest)
    print('zig: installed')


def install_release_binary(name, url, tmp):
    dest = TOOLS / 'bin' / name
    if dest.is_file():
        print(f'{name}: present')
        return
    archive = tmp / url.rsplit('/', 1)[1]
    download(url, archive)
    with tarfile.open(archive) as tf:
        tf.extractall(tmp / name, filter='data')
    found = next(p for p in (tmp / name).rglob(name) if p.is_file())
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(found, dest)
    dest.chmod(0o755)
    print(f'{name}: installed')


def install_onnxruntime(platform_id, suffix, tmp):
    dest = TOOLS / 'onnxruntime' / platform_id
    if (dest / 'LICENSE').is_file():
        print(f'onnxruntime {platform_id}: present')
        return
    with urllib.request.urlopen(f'https://pypi.org/pypi/onnxruntime/{ORT_VERSION}/json') as response:
        files = json.load(response)['urls']
    wheel = next(f for f in files if '-cp312-' in f['filename'] and f['filename'].endswith(suffix))
    archive = tmp / wheel['filename']
    download(wheel['url'], archive)
    if sha256(archive) != wheel['digests']['sha256']:
        sys.exit(f'{wheel["filename"]} does not match its PyPI checksum.')
    staging = tmp / f'ort-{platform_id}'
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(staging)
    package = staging / 'onnxruntime'
    dest.mkdir(parents=True)
    for item in list((package / 'capi').iterdir()) + [package / 'LICENSE', package / 'ThirdPartyNotices.txt']:
        if item.is_file() and 'pybind' not in item.name and item.suffix != '.py' and (item.name.startswith(('onnxruntime', 'libonnxruntime')) or item.parent == package):
            shutil.copy2(item, dest / item.name)
    (dest / 'source.json').write_text(json.dumps({'wheel': wheel['filename'], 'sha256': wheel['digests']['sha256']}, indent=2) + '\n', encoding='utf-8')
    print(f'onnxruntime {platform_id}: installed')


def main():
    if platform.system() != 'Darwin':
        sys.exit('Cross-building is set up for macOS hosts: a Mac can build all three bundles, Linux and Windows cannot build the Mac one.')
    llvm = Path(subprocess.run(['brew', '--prefix', 'llvm@22'], capture_output=True, text=True).stdout.strip() or '/nonexistent') / 'bin'
    missing = [tool for tool in LLVM_TOOLS if not (llvm / tool).is_file()]
    if missing:
        sys.exit(f'Missing {", ".join(missing)}. Install LLVM with: brew install llvm@22')
    version = rustc_version()
    arch = {'arm64': 'aarch64', 'x86_64': 'x86_64'}[platform.machine()]
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for target in CROSS_TARGETS:
            install_rust_std(version, target, tmp)
        install_zig(tmp)
        install_release_binary('cargo-zigbuild', f'https://github.com/rust-cross/cargo-zigbuild/releases/download/v{ZIGBUILD_VERSION}/cargo-zigbuild-{arch}-apple-darwin.tar.xz', tmp)
        install_release_binary('cargo-xwin', f'https://github.com/rust-cross/cargo-xwin/releases/download/v{XWIN_VERSION}/cargo-xwin-v{XWIN_VERSION}.universal2-apple-darwin.tar.gz', tmp)
        for platform_id, suffix in ORT_WHEELS.items():
            install_onnxruntime(platform_id, suffix, tmp)
    print('Cross tools ready. Build with: uv run code/build.py --target linux-x64 (or windows-x64) --model MODEL --output BUNDLE')


if __name__ == '__main__':
    main()
