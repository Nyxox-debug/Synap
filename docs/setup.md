# Setup

Build and install instructions for Synap.

---

## Requirements

- **C++17** or later
- **CMake** 3.18+
- **Python** 3.10+ (3.13 recommended)
- **pybind11** (fetched automatically via CMake FetchContent or available in `extern/`)

---

## Clone

```bash
git clone https://github.com/Nyxox-debug/Synap.git Synap
cd Synap
```

---

## Python Environment

Create a virtual environment before building so CMake can find the correct Python interpreter:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Build

Synap uses CMake presets. The default preset configures a release build with pybind11 bindings.

```bash
cmake --preset default
cmake --build build
```

After a successful build, the compiled `synap` module (`.so` on Linux/macOS, `.pyd` on Windows) will be located inside `build/`.

---

## Making the Module Available


### Option A — Add to PYTHONPATH (Recommended)

> **NOTE:** I use `direnv` for automatic environment variable setup.

### `.envrc`

```bash
# .envrc
export PYTHONPATH=$PWD/build
source $PWD/venv/bin/activate  # adjust path if your venv folder is different
````
Then run direnv allow

or

Run this in your temrinal

```bash
export PYTHONPATH="$PWD/build:$PYTHONPATH"
```

Add that line to your shell rc file (`~/.bashrc`, `~/.zshrc`) to make it permanent. (Not recommended)

### Option B — Run from the build directory

```bash
cd build
python3 -c "import synap; print('ok')"
```


### Option C — Editable install (if a `setup.py`/`pyproject.toml` is added later)

```bash
pip install -e .
```

---

## CMakePresets.json

The included `CMakePresets.json` defines a `default` configure preset. You can inspect or extend it for debug builds:

```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "setup",
      "generator": "Unix Makefiles",
      "binaryDir": "${sourceDir}/build",
      "cacheVariables": {
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "CMAKE_CXX_STANDARD": "17",
        "CMAKE_SUPPRESS_DEVELOPER_WARNINGS": "1"
      }
    }
  ]
}
```

For a debug build, duplicate the preset and set `CMAKE_BUILD_TYPE` to `Debug`.

---

## Type Stubs

Python type stubs for the `synap` module live in `stubs/synap.pyi`. To make editors (e.g., VS Code with Pylance) pick them up, the `pyrightconfig.json` at the project root points to this directory:

```json
{
  "stubPath": "stubs"
}
```

No extra configuration is needed — open the project root in your editor and type checking should work out of the box.

---

## Running Examples

With the module on your path, run the gradient descent demo from the project root:

```bash
python3 python/test_grad_descent.py
```
---

## Troubleshooting

**`ModuleNotFoundError: No module named 'synap'`**
The `.so` is not on `PYTHONPATH`. Either run from the `build/` directory or set the variable as shown above.

**CMake can't find Python**
Make sure your virtual environment is activated before running `cmake --preset default`. CMake uses `find_package(Python3)` and will pick up the active interpreter.

**pybind11 not found**
Check that `extern/pybind11` exists (if vendored) or that `FetchContent` has network access during the configure step.
