# DecisionGator in one file

`decisiongator_standalone.dll` (Windows), `libdecisiongator_standalone.so` (Linux) or `libdecisiongator_standalone.dylib` (Mac) is the whole DecisionGator component in a single library: the model, tokenizer, manifest, and ONNX Runtime are all inside it. Put it next to your program and call it. Nothing else has to sit beside it.

It has the same C functions as the normal bundle (`decisiongator.h`), so the C#, C, Rust, and Go examples work unchanged; the C# wrapper looks for the one-file library first and falls back to the normal bundle. The `notices/` folder holds the licenses for the parts inside, which you must pass on when you give the library to someone else. The program never reads them.

Compared with the normal bundle:

- About 610 MB as one file instead of a folder of about 600 MB.
- Answers are the same; the first call is faster because the model's checksum is checked when the library is built rather than every time it loads.
- Replacing the model means rebuilding the library.
- `dg_load` still loads a separate bundle folder, using the ONNX Runtime built into the library.
- Windows: needs the Microsoft Visual C++ 2015–2022 x64 Redistributable (`winget install --id Microsoft.VCRedist.2015+.x64 --exact`) and Windows 10 version 1903 or newer, because the built-in ONNX Runtime links to `DirectML.dll`, which ships with Windows from that version on.

## Getting it

The Windows x64 build is published with the other release files: `uv run code/fetch_released.py --only standalone-windows-x64` puts it in `released/standalone-windows-x64/`.

## Building it

```text
uv run code/fetch_released.py --only windows-x64     # or macos-arm64, linux-x64
uv run code/setup.py                                 # project Rust toolchain, once
uv run code/build_standalone.py
```

The result goes to `released/standalone-<platform>/`. `code/standalone/` is a separate Rust package that compiles the same source as the normal library (`../src/lib.rs`) with the `dg_embedded` setting, which loads the model from inside the library instead of from files beside it. Its build script checks the model and tokenizer against the manifest's hashes before compiling them in. The model is attached with the assembler's `.incbin` because `include_bytes!` on a 578 MB file runs the compiler out of memory. ONNX Runtime comes from the `ort` crate's prebuilt static library, downloaded on the first build; on Windows it needs MSVC 14.44 (Visual Studio 2022 version 17.14) or newer to link.

Only the Windows x64 build has been built and tested (Windows 11, October 2026). The Mac and Linux code paths compile the same way in principle but have not been run.
