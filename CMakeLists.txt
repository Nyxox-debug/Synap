cmake_minimum_required(VERSION 3.20)
project(SimpleCppPython)

# Find pybind11 (installed via pip or system-wide)
find_package(pybind11 REQUIRED)


# Create Python module
pybind11_add_module(example
    src/bindings.cpp
    src/tensor/tensor.cpp
)
