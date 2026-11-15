# Examples you can run

Each folder holds one small program that makes the same four calls: a yes/no answer, the probability of yes, a stricter answer with criteria, and a choice among three teams. Every one prints:

```text
urgent: true
p_yes = 0.915
confident urgent: true
billing            0.879
sales              0.069
technical support  0.052
```

Numbers can differ slightly between computers and in the browser (see [why](../reports/accuracy-v4.1.md#answers-can-differ-slightly-between-platforms)).

## 1. Get the repository and the model

The model files are too large for GitHub, so a script downloads just what your language needs and checks every file. It needs [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
git clone https://github.com/CarlFreeAiEngineer/DecisionGator.git
cd DecisionGator
uv run code/fetch_released.py --for java      # or python, c, rust, go, csharp, node, browser
```

The download is about 600 MB. Prebuilt files exist for Apple silicon Macs, Linux x64 and Windows x64; the browser example works anywhere.

## 2. Run your language

Run these from the repository folder.

| Language | You need | Run |
| --- | --- | --- |
| Python | uv | `uv run examples/python/example.py` |
| Java | JDK 17+ and Maven | `cd examples/java` then `mvn -q compile exec:java` |
| Node.js | Node.js 24+ | `node examples/node/example.mjs` |
| C# | .NET 8+ SDK | `cd examples/csharp` then `dotnet run` |
| Rust | Rust (cargo) | `cd examples/rust` then `cargo run --release` |
| Go | Go 1.21+ and a C compiler | `cd examples/go` then `go run .` |
| C | a C compiler | see below |
| Browser | Node.js, a browser | `node examples/browser/serve.mjs`, then open the address it prints |

The Node.js example installs the package for your computer from `released/node/` the first time it runs. The Java example gets JNA, the library Java uses to call native code, from Maven Central.

C on a Mac:

```sh
cd examples/c
cc example.c -I../../code/include -L../../released/macos-arm64 -ldecisiongator -Wl,-rpath,@executable_path/../../released/macos-arm64 -o example
./example
```

C on Linux:

```sh
cd examples/c
cc example.c -I../../code/include -L../../released/linux-x64 -ldecisiongator -Wl,-rpath,'$ORIGIN/../../released/linux-x64' -o example
./example
```

C on Windows, in a Visual Studio Developer Command Prompt:

```bat
cd examples\c
cl /nologo example.c /I..\..\code\include ..\..\released\windows-x64\decisiongator.lib
set PATH=%CD%\..\..\released\windows-x64;%PATH%
example.exe
```

On Windows the Go and Rust examples also need `released\windows-x64` on `PATH` when they run, and the native examples need the Microsoft Visual C++ 2015-2022 x64 Redistributable.

## Using it in your own project

These examples point at this checkout's `released/` folder. In your own project, follow the guide for your language: [Python](../released/python/README.md), [Java](../java/README.md), [Node.js](../javascript/README.md), [browser](../web/README.md), [C#](../csharp/README.md), [C](../code/README.md). [EXAMPLE_USAGE.md](../EXAMPLE_USAGE.md) shows every call in every language.

## Experimental: the evidence check

`evidence_check.py` runs the separate evidence-check model: which sentences of a reply does the text it was written from not support? `uv run examples/evidence_check.py --demo` downloads the model (about 600 MB) on first run. It catches only about half of real inventions so far; see [its notes](../released/evidence-check-0.1/README.md).
