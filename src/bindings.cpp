#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor/tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(example, m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<size_t>, bool>(),
             py::arg("shape"),
             py::arg("requires_grad") = false)
        .def("shape", &Tensor::shape)
        .def("clone", &Tensor::clone)
        .def("view", &Tensor::view)
        .def("zero_grad", &Tensor::zero_grad);
}
