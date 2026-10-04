# DecisionGator for C# and .NET

Your software probably needs this.

```csharp
using DecisionGator;

if (Decisions.IsYes("Can we meet on Friday?", "Is the user asking for an appointment?"))
    ShowAvailableTimes();
```

`csharp/DecisionGator/` is a small .NET 8 class library that calls the native component through P/Invoke. No C++ projects, no glue code to compile, no server. Applications reference the project (or a build of it) and ship a native bundle beside their program.

## Getting the native bundle

The library, model, tokenizer, and manifest are one folder per platform, fetched with `uv run code/fetch_released.py --only macos-arm64` (or `linux-x64`, `windows-x64`) into `released/`. At run time the wrapper looks for the native library in this order:

1. `Bundle.Directory`, if your code sets it before the first call.
2. The `DECISIONGATOR_BUNDLE` environment variable.
3. A `decisiongator` folder next to the application.
4. The application folder itself.

In each place it first looks for the one-file library, `decisiongator_standalone.dll` (or `libdecisiongator_standalone.so` / `.dylib`), which has the model built in; see [code/standalone](../code/standalone/README.md). With that file next to your executable there is nothing else to copy.

For deployment, copy the whole bundle folder into a `decisiongator` directory next to your executable and nothing needs configuring. The library is loaded by absolute path so it can find its model files beside itself.

## API

| Call | Returns |
| --- | --- |
| `Decisions.IsYes(content, question, criteria?, threshold = 0.5)` | `bool`, true when p(yes) >= threshold |
| `Decisions.IsYesP(content, question, criteria?)` | `double` from 0 to 1 |
| `Decisions.Choose(content, question, options, criteria?, threshold = 0)` | index of the best option, or -1 when below threshold |
| `Decisions.ChooseP(content, question, options, criteria?)` | every option as `Choice(Index, P)`, best first, probabilities summing to one |

`Criteria` is a record with `Yes` and `No` strings describing what counts as each answer. Native failures throw `DecisionGatorException` with a `StatusCode`; invalid thresholds throw `ArgumentOutOfRangeException`. An error is never returned as `false` or `-1`.

The first call loads the bundle and takes a few seconds; later calls reuse it and take about a tenth to a third of a second on a laptop CPU with the 0.4.1 model (see [the 0.4.1 report](../reports/accuracy-v4.1.md)). Calls are safe from any thread and run one at a time. Keep the component loaded for the life of the process; do not call it from process-exit handlers.

## Run the smoke test

Install the .NET SDK with your package manager (`brew install --cask dotnet-sdk`, `winget install Microsoft.DotNet.SDK.8`, or your distribution's package), then:

```text
cd csharp/Smoke
DECISIONGATOR_BUNDLE=../../released/macos-arm64 dotnet run
```

On Windows, set the variable to `..\..\released\windows-x64` first. The output shows a refund yes/no pair, a criteria-and-threshold call, a `Choose` ranking, and an error reported as an exception. The library targets .NET 8; the smoke app rolls forward to whatever newer runtime is installed.

Tested on macOS arm64 with .NET SDK 10.0.401 and the 0.4.1 Mac bundle (about 620 MB; the smoke program's refund, criteria, choice, and error cases all behave as expected). Linux and Windows use the same code and their own bundles but have not been exercised here yet. A NuGet package is not published; the platform bundles are far over nuget.org's size limit, so the intended packaging is the project reference plus a bundle folder shipped with the application.
