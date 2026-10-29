# Development and builds

The Rust library, portable toolchain setup, and native build helper are implemented for Mac arm64, Linux x64/glibc, and Windows x64/MSVC. All three builds have passed native and packaged inference checks. See [the release inventory](../released/README.md) for exact tested systems and prerequisites, and [native build commands](../code/README.md) to reproduce a build.

## Languages and layout

Implement the native component in Rust, exposing the C interface in [runtime.md](runtime.md). Python remains the training and experiment language, run through `uv`; applications using the native component will not need Python or a Rust installation.

- `code/` contains `Cargo.toml`, `Cargo.lock`, Rust source, and local `target/` build output.
- `code/setup.py` installs or copies the pinned project-local Rust toolchain; `code/build.py` builds and tests a native bundle.
- `tools/rust/` contains the platform-specific Rust toolchain, including `bin/cargo` and `bin/rustc` (with `.exe` on Windows).
- `tools/cargo-home/` contains Cargo's downloaded crates and cache.
- `released/` holds the documented release bundle layout. Its README states which targets are planned and which have actually shipped.

Build the library as a Rust `cdylib`, the output intended for loading from other languages, with explicit C exports. Keep Rust types and panic unwinding inside the library boundary. The C header and Python wrapper must agree with the native interface. See the [Rust linkage reference](https://doc.rust-lang.org/reference/linkage.html).

## Portable local tools

Follow the layout inspected in KitchenSync's `code/build.py` and `specs/DEVELOPMENT.md`: use a local Rust toolchain under `tools/rust/`, put its `bin/` first on the build process's `PATH`, and set `CARGO_HOME` to this project's `tools/cargo-home/`. DecisionGator must be self-contained; do not link to or depend on KitchenSync's installed tools.

Pin the compiler version and record per-platform toolchain sources and checksums when implementing setup. Keep tools and caches out of Git. Each machine obtains its own matching toolchain; portable here means local to the checkout, not that one toolchain binary works on every OS. Do not change the user's global compiler or shell configuration.

The build helper must locate the checkout from its own file path, invoke its local Cargo explicitly, verify the host OS and architecture, and build with the committed dependency lockfile (`--locked`). Report missing tools clearly instead of silently using a global compiler. Keep build output under `code/target/` and copy only the tested bundle for the current platform into `released/`.

Rust still needs the platform's linker and SDK where applicable. Document and check these prerequisites, including Apple command-line developer tools on macOS and MSVC build tools on Windows. Keep additional portable dependencies under `tools/` where practical. A local Rust installation alone is not a promise of a complete standalone OS build environment.

## Three release targets

| Platform | Rust target | Output directory |
| --- | --- | --- |
| Windows x64 | `x86_64-pc-windows-msvc` | `released/windows-x64/` |
| Mac M1+ | `aarch64-apple-darwin` | `released/macos-arm64/` |
| Linux x64, glibc | `x86_64-unknown-linux-gnu` | `released/linux-x64/` |

The target names follow [Rust's platform documentation](https://doc.rust-lang.org/rustc/platform-support.html). Build and test natively on each platform; cross-compilation is not a requirement. Linux delivery is an ELF shared library (`.so`), with the minimum glibc version and tested distributions recorded before release. A `.so` suffix alone does not guarantee compatibility with every Linux installation.

Follow the accuracy gates in [evaluation.md](evaluation.md), independently of packaging. Freeze cross-platform reference outputs with `tests/platform_parity.py`; a new binary must match the reference weights/tokenizer and pass probability and boolean comparisons. `tests/package_check.py` builds a C consumer, relocates the bundle, removes development tools from its runtime PATH, and tests inference and clean exit. It verifies network denial on Mac and on Linux when user/network namespaces are available. On Windows, `tests/windows_offline_check.py` requires an elevated MSVC shell and verifies a temporary outbound firewall block for its own C executable, inference while blocked, and connectivity recovery after removing the rule.

The Linux build machine used for this release was emeraldslate, running Omarchy Linux. That is no longer true: its disk was wiped on 2026-10-26 and it now runs Windows 11 Pro, with no Linux, no Docker and no VMs, so the Omarchy Linux host, the `omarchy-windows` Windows guest and the `ssh ace@emeraldslate` login are gone. A new Linux build machine is not set up yet. EmeraldSlate (`ssh human@emeraldslate`, `vnc://EmeraldSlate`) could serve for native Windows builds if set up. Machine addresses and credentials are not part of the software component and are not required by its build helpers.

Build Python wheels with `uv run code/build_python.py --target PLATFORM`. `uv run code/remote_packages.py` assembles and tests Python and Node packages on the matching host after its native bundle is ready. Development tools and build-time downloads stay out of consumer inference. Cross-platform packaging of existing assets is allowed; native code and its execution checks still run on the target OS.

Before calling a platform supported, load its packaged library from a C host and the Python binding, run real model evaluations offline, check agreement with the training pipeline, and measure resource use. Test from a clean location without the development toolchain on `PATH`; resolve bundled dependencies relative to the installed bundle. Publish minimum system requirements and required system dependencies.

Each bundle must include the assets and manifest specified in [runtime.md](runtime.md), plus its header, linkage files, notices, and integration example. Record source revision, compiler and dependency versions, model identity, and verification results so a binary can be traced back to its inputs. A build on one platform must not overwrite another platform's bundle.

## Hosting comes later

Defer Git LFS, GitHub release assets, and website hosting decisions until a useful version runs locally. Do not introduce large-file storage or publishing infrastructure now. Preserve this release layout whichever distribution method is eventually chosen.

## Java packaging

Java 17 sources and a Maven build live under `java/`. Run `uv run java/build.py --install` with the project-local JDK and Maven to build, test, package, and install the API and platform bundle into `tools/maven-repository`. Run `uv run java/check_bundle.py` to verify JAR-only offline inference on the Mac. No artifacts have been published to Maven Central. See [the Java guide](../java/README.md) for tool setup and dependency examples.

## JavaScript and browser builds

The Node-API addon and JavaScript/TypeScript interfaces live in `javascript/`. From that directory, run `npm ci`, `npm run build`, `npm test`, and `npm run pack:release` with Node 24. The build uses the matching native bundle from `released/`, compiles the addon, and stages a platform tarball under `released/node/`. Consumers install that tarball without a compiler or runtime download. `node javascript/test/consumer.mjs` verifies a fresh installed package. See [the Node guide](../javascript/README.md) for compiler prerequisites, overrides, and platform build commands.

Browser sources live in `web/`. From the project root, run `npm ci --prefix web`, `node web/build.mjs`, and `node web/tests/tokens.mjs`. This produces the self-contained browser component and demo in `released/web/`, using the same weights through ONNX Runtime Web. Use Node 22 or 24 for browser test tooling. Install test browsers with `node web/node_modules/playwright/cli.js install chromium firefox webkit`, then run `node web/tests/browser.mjs chromium firefox webkit`. Tests use temporary browser profiles, compare native outputs, and verify cached offline execution.

Build the browser release before packaging Node if the combined package should include `decisiongator/web`. Browser hosting must preserve worker and asset paths or supply the documented overrides. Read [JavaScript architecture and verification](javascript-and-webassembly.md) and [browser deployment instructions](../web/README.md).
