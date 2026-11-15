# DecisionGator for Python

Fetch the wheels with `uv run code/fetch_released.py --only python` if they are not already here, then install the wheel for your platform into your application. It includes the native component, weights, tokenizer, and license notices. Inference needs no Python dependencies, downloads, server, or development tools.

From this checkout, add the Mac package to a Python project:

```console
uv add /path/to/DecisionGator/released/python/decisiongator-0.4.1-py3-none-macosx_14_0_arm64.whl
```

Then use it normally:

```python
from decisiongator import is_yes

if is_yes("Please send me a replacement lid.", "Is a replacement part requested?"):
    arrange_replacement()
```

The wheel supports Python 3.11 and later. Python 3.12 is tested on Mac, Linux, and Windows. The Mac wheel conservatively targets macOS 14 and later on Apple silicon: its bundled runtime requires macOS 13.3, which ordinary major-version compatibility tags cannot express precisely. The Linux wheel uses a plain `linux_x86_64` tag, not a claim of manylinux compatibility; consult the native bundle's system requirements. The Windows wheel is tested on Windows 11 x64 and requires the Microsoft Visual C++ 2015-2022 x64 Redistributable.

Build a wheel from an existing native bundle with `uv run code/build_python.py --target macos-arm64` (or `linux-x64` / `windows-x64`). The helper verifies manifest checksums and includes the existing assets without recompiling or retraining. The application wheel contains only the runtime interface; training tools remain in the source checkout.

Run `uv run tests/python_wheel_check.py PATH_TO_WHEEL` on the matching platform. On macOS the test denies networking and access to this checkout, installs into an isolated consumer, and checks concurrent initialization, ordinary calls, criteria, thresholds, and errors. See [the test report](../../reports/python-wheel.json).

These packages are local release artifacts. Nothing has been uploaded to PyPI.

Version 0.4.3 wheels for Mac and Linux x64 (`decisiongator-0.4.3-py3-none-macosx_14_0_arm64.whl`, `decisiongator-0.4.3-py3-none-linux_x86_64.whl`) hold the same native library as 0.4.1 with a model trained further for [the guard check](../../specs/guard-check.md); see [its report](../../reports/guard-check.md). They were built here with `code/rebundle.py` and `code/build_python.py --version 0.4.3` and are not yet on the download server. There is no Windows 0.4.3 wheel.
