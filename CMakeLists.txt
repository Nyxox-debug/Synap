cmake_minimum_required(VERSION 3.20)
project(SimpleCppPython)

# Find pybind11 (installed via pip or system-wide)
find_package(pybind11 REQUIRED)


# Create Python module
pybind11_add_module(synap
    src/bindings.cpp
    src/tensor/tensor.cpp
    src/tensor/autodiff.cpp
)

# find_package(Python3 REQUIRED COMPONENTS Interpreter)
#
# add_custom_target(
#     synap_stubs
#     COMMAND
#         ${Python3_EXECUTABLE} -m pybind11_stubgen synap
#         --output-dir ${CMAKE_CURRENT_SOURCE_DIR}/stubs
#     DEPENDS synap
# )
