# Release bundles

This directory holds installable local DecisionGator releases. Every target is meant to use the same weights, tokenizer, input rules, and calibration. Packaging does not improve or reduce decision accuracy, though the quantized 0.4.1 weights round slightly differently on each processor, so borderline cases can differ between platforms. Version 0.4.1 answered 435/480 held-out yes/no cases, 74/80 fresh cases, and 18/20 held-out multiple-choice cases correctly on the Mac. See [the 0.4.1 report](../reports/accuracy-v4.1.md).

**Version status by platform.** All three native bundles (Mac, Windows x64, Linux x64), their Python wheels, Node tarballs, and Java classifier JARs, and the browser assets are version 0.4.1: the 0.4.0 DeBERTa-v3-large model stored as 8-bit and 4-bit integers, using one thread per fast core. All were built on 23 September 2026 from the same model file. The frozen cross-platform comparison cases must agree within the tolerance in each manifest (probability difference at most 0.25, changed decisions on at most 5% of cases).

Native bundles:

| Platform | Bundle directory | Native library |
| --- | --- | --- |
| Windows x64 | `windows-x64/` | `decisiongator.dll` |
| Mac M1 and newer (Apple silicon) | `macos-arm64/` | `libdecisiongator.dylib` |
| Linux x64, glibc | `linux-x64/` | `libdecisiongator.so` (ELF) |

Each native bundle contains the component, its CPU runtime, weights, tokenizer, C header, checksums, and license notices. Windows packaging also includes the import library for C applications. No separate inference service or consumer-side compiler is needed.

The Mac build is tested on Apple silicon with macOS 26.6.2. Binary metadata requires macOS 13.3 or later; the Python wheel conservatively targets macOS 14+. The Linux build is tested on emeraldslate, an Omarchy Linux x64 machine with glibc 2.44. Its symbols require glibc 2.34+, GLIBCXX 3.4.22+, and CXXABI 1.3.11+, plus the system libraries listed in `linux-x64/system-dependencies.json`. Those minimum symbol versions do not mean every older distribution has been tested. The Windows DLL is built and tested on Windows 11 x64, build 26200, in emeraldslate's existing `omarchy-windows` guest. It requires the Microsoft Visual C++ 2015-2022 x64 Redistributable. Older Windows releases have not been tested.

Build each native bundle on its own platform with [the portable build helper](../code/README.md). Alpine/musl, Linux ARM, Intel Macs, Windows ARM, and mobile devices are outside this release's tested scope.

The [Windows verification report](../reports/windows-release.md) records native, Python, Node, and Java checks, cross-platform agreement, and deployment requirements.

## Language packages

- [Python wheels](python/README.md) include all native assets and have no Python runtime dependencies.
- [Java JARs](../java/README.md) provide an ordinary Java 17 API, platform classifier bundles, sources, and Javadoc. JNA is a normal transitive dependency.
- [Node.js packages](../javascript/README.md) include a Node-API addon, JavaScript/TypeScript interfaces, native assets, and the browser entry. Install the tarball matching the operating system and architecture.
- [Browser assets](../web/README.md) include the WebAssembly runtime, worker, declarations, weights, tokenizer, and an offline-capable example. Copy `web/` into your application's assets. Chromium, Firefox, and WebKit engines passed real inference and offline tests on the Mac.

The 0.4.1 browser bundle is approximately 580 MB and each 0.4.1 native bundle approximately 600 to 620 MB (weights 578 MB). A running process holds about 1.5 GB. The published 0.4.0 bundles (float16, 872 MB weights, about 5 GB in memory) remain downloadable with `--version 0.4.0`. The previous 0.3.0 Mac bundle (229 MB) is preserved at `models/macos-arm64-v3/` for anyone who prefers size over accuracy. The combined npm tarball contains separate native and browser assets, including two copies of the weights, so it is larger. Browser use requires the web app's initial asset delivery and sufficient storage for offline installation. See [browser measurements and limitations](../reports/browser-webassembly.md).

The binaries in this directory are not committed to Git; only the `.md` and `.json` records are. Published versions live at `https://62-84-178-253.sslip.io/DecisionGator/files/<version>/` with a `SHA256SUMS.txt`. `uv run code/fetch_released.py` downloads and verifies them into this directory (`--only macos-arm64` and similar select parts). Maintainers publish a new version with `uv run code/publish_released.py --version X.Y.Z`, which writes the checksum file and copies this directory to the server.

The experimental evidence check, a separate model, is published as `evidence-check-0.1`; see [its notes](evidence-check-0.1/README.md). `examples/evidence_check.py` downloads it on first run.

No artifacts have been published to PyPI, Maven Central, or npm. Local installation and packaging are implemented independently of that future publishing decision.
