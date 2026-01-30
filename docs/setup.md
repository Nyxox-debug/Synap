# Building Pybind11 Project on Linux

## 1. Install pybind11 system-wide

For Debian/Ubuntu-based systems:

```bash
sudo apt install pybind11-dev
````

For Arch Linux:

```bash
sudo pacman -S pybind11
```

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

> I use `direnv` for automatic environment variable setup.

### `.envrc`
```bash
export PYTHONPATH=$PWD/build
````

For setup of Stub for lsp:

> **Note:** Use the name of the module used in the C++ file.

```bash
pybind11-stubgen example
```
