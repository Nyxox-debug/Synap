# Building Pybind11 Project on Linux

## 1. Install pybind11 system-wide

You don't need to I updated to use git submodule so pybind11 is at `/extern/pybind11`

## 2. Configure the project

Use CMake presets to configure:

```bash
cmake --preset setup
```

## 3. Build the project

```bash
cmake --build build
```

or alternatively, enter the build directory and run `make`:

```bash
cd build
make
```

## 4. Lsp Configuration although I already have the stubs at `/stubs`

For setup of Stub for lsp:

> **Note:** Use the name of the module used in the C++ file.

```bash
pybind11-stubgen synap 
```

I use pyright, so i configured the stubPath
`pyrightconfig.json`

```json
{
  "typeCheckingMode": "basic",
  "stubPath": "./stubs",
  "extraPaths": [
    "./build",
    "./build/stubs/"
  ]
}
```

> **NOTE:** I use `direnv` for automatic environment variable setup.

### `.envrc`

```bash
# .envrc
export PYTHONPATH=$PWD/build
source $PWD/venv/bin/activate  # adjust path if your venv folder is different
````

> **NOTE:** in the pyfile use help(synap) for information concering synap library
