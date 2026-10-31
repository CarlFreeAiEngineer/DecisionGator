# Native builds

A Mac can build all three native bundles (Mac arm64, Linux x64 with glibc, Windows x64); see [Build all three on a Mac](#build-all-three-on-a-mac). Linux and Windows machines can build only their own bundle. Either way, each bundle's checks must run on its own platform: a bundle built on a Mac for another platform stays marked pending until `tests/target_check.py` passes there.

Use an existing exported model directory containing `manifest.json`, `model.onnx`, and `tokenizer.json`. Training and PyTorch are unnecessary on build machines. The build verifies both asset hashes before compiling.

## Prerequisites

- Install `uv` and `rustup` with the machine's package manager, or provide a matching existing Rust toolchain to the setup command below.
- Mac: Apple command-line tools.
- Linux: a C compiler and linker, typically the distribution's `build-essential` package, plus `binutils` for `readelf`. Build on the oldest glibc environment you intend to support; the recorded build-host glibc version is not a measured minimum.
- Windows: Visual Studio Build Tools with the C++ workload and Windows SDK, plus the Microsoft Visual C++ 2015–2022 x64 Redistributable required by ONNX Runtime. Run the build in an x64 Native Tools shell. The Rust DLL and ONNX Runtime both rely on normal Windows system/runtime libraries.

Set up Rust 1.98.1 inside this checkout:

```text
uv run code/setup.py
```

The helper uses rustup with project-local homes, then copies the matching toolchain into `tools/rust/`. It does not change global compiler settings. To copy an already installed matching toolchain instead:

```text
uv run code/setup.py --rust-from PATH_TO_TOOLCHAIN
```

The directory must contain `bin/rustc` and `bin/cargo` (`.exe` on Windows). Existing project tools are never overwritten. Downloaded tools and Cargo crates stay under `tools/`.

## Build all three on a Mac

Once per checkout, after `uv run code/setup.py`, install Homebrew LLVM and the cross tools:

```text
brew install llvm@22
uv run code/setup-cross-tools.py
```

This adds the Rust standard libraries for Linux and Windows, Zig with cargo-zigbuild (the C compiler and linker for Linux), cargo-xwin (Visual Studio-style Windows DLLs), and ONNX Runtime 1.22.1 for Linux and Windows taken from the official Python wheels, all under `tools/`. LLVM's `clang-cl` and `llvm-lib` compile the tokenizer's C code for Windows. On first use cargo-xwin downloads Microsoft's C runtime and Windows SDK libraries (about 1 GB) into `tools/xwin`; using them means accepting Microsoft's license terms.

```text
uv run --python 3.12 --with onnxruntime==1.22.1 code/build.py --model MODEL_DIRECTORY --output released/macos-arm64
uv run code/build.py --target linux-x64 --model MODEL_DIRECTORY --output released/linux-x64
uv run code/build.py --target windows-x64 --model MODEL_DIRECTORY --output released/windows-x64
```

Each cross build takes about a minute. It runs the Rust unit tests on the Mac, builds the library and a C smoke program for the target, records dependencies with LLVM's readers, and writes `build-checks.json` with status `pending`. The Linux library targets glibc 2.28 and has a soname, so the smoke program finds it through `$ORIGIN`. Then, on a machine of that platform with a checkout of this repository and `uv` (no compiler needed), copy the bundle over and run:

```text
uv run tests/target_check.py --bundle BUNDLE_DIRECTORY
```

It verifies every file hash, runs the smoke program, the native ABI and Python checks, and the frozen platform comparison when it applies, then marks the bundle passed. Copy the checked `build-checks.json` and `platform-parity.json` back into the Mac's copy of the bundle. For the 0.4.1 model, cross-built Linux and Windows bundles gave exactly the same comparison results as the bundles built on those platforms, since the inference itself runs in the same ONNX Runtime files.

## Build without training dependencies

Choose a fresh output directory; the helper preserves existing bundles. These commands use a small isolated uv environment with ONNX Runtime, not the project's training group:

```text
uv run --python 3.12 --with onnxruntime==1.22.1 code/build.py --model MODEL_DIRECTORY --output released/macos-arm64
uv run --python 3.12 --with onnxruntime==1.22.1 code/build.py --model MODEL_DIRECTORY --output released/windows-x64
uv run --python 3.12 --with onnxruntime==1.22.1 code/build.py --model MODEL_DIRECTORY --output released/linux-x64
```

When you pack a checkout or bundle on a Mac to build on another machine, create the archive with `COPYFILE_DISABLE=1 tar ...`. Otherwise macOS adds hidden `._name` metadata files; they end up in the bundle's `notices/` folder and in the Windows wheel, and the Java bundle check then fails with `Missing bundled asset: notices/._...`. Each release needs roughly 4 GB free on the build machine (the Node consumer test unpacks a 1.2 GB package); the Windows guest's small disk fills up unless earlier bundles and package files are deleted first.

Use only the command matching the current host. On Windows, `code/windows_build.ps1 -Root C:\path\to\checkout` runs the whole sequence (MSVC environment, Rust setup, native build, relocation check, Python and Node packages) from an ordinary shell; its header explains how to start it detached over SSH. For an already downloaded official ONNX Runtime 1.22.1 CPU package, pass `--runtime-dir EXTRACTED_PACKAGE_DIRECTORY` and omit `--with onnxruntime==1.22.1`. The directory must contain the library under `lib/` (or `capi/`) and its `LICENSE` and `ThirdPartyNotices.txt`. An installed `onnxruntime` Python package directory also works. The helper checks the runtime's actual version and that it loads on this machine.

The build runs Rust tests, compiles the release library and a C host, then runs real inference and the native/Python checks. For the released model, it also checks the frozen platform comparison cases. A different model gets an explicit “not applicable” report until a matching reference is frozen. Only successful completion produces `build-checks.json`. A failed build may leave an incomplete output directory; inspect it and use a fresh destination for a retry. A separate network-denial check is still needed to establish offline behavior on a newly supported platform.

`system-dependencies.json` records linked libraries and binary compatibility metadata: Mac minimum OS versions, Linux GLIBC/GLIBCXX/CXXABI symbol requirements, or Windows DLL dependencies. These are compatibility floors, not evidence of testing on older operating systems. The Mac binaries inspected during development require macOS 13.3 because of ONNX Runtime; a wrapper supporting an older OS does not lower the bundle's requirement.

Windows bundles include `decisiongator.dll`, `decisiongator.lib` for C linking, and `onnxruntime.dll`. Linux bundles contain `libdecisiongator.so` and `libonnxruntime.so`; Mac bundles contain the corresponding `.dylib` files. All include the model, tokenizer, header, manifest, notices, and C smoke host. Runtime dependencies are copied when present. Loading the native component by absolute path allows automatic discovery of adjacent model files regardless of the application's working directory.

Do not dynamically unload the DecisionGator library: it retains its default session until process exit. Advanced session handles can be released and reloaded normally. Shut down inference threads before process exit and do not call DecisionGator from exit hooks.

On Windows, the operating system reclaims the default session and ONNX Runtime environment when the process terminates. The library deliberately avoids an `atexit` cleanup callback on Windows: DLL detach can occur after ONNX Runtime has detached and worker threads have stopped. Unix retains its explicit session/environment cleanup before runtime teardown. This difference does not affect `dg_release`, which releases an explicit session during normal execution.
