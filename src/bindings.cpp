#include "tensor/autodiff.h"
#include "tensor/tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(synap, m) {
  py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
      .def(py::init<std::vector<size_t>, bool>(), py::arg("shape"),
           py::arg("requires_grad") = false)
      .def("shape", &Tensor::shape)
      .def("clone", &Tensor::clone)
      .def("view", &Tensor::view)
      .def("zero_grad", &Tensor::zero_grad)
      .def_readonly("requires_grad", &Tensor::requires_grad)
      .def_readonly("grad", &Tensor::grad);

  py::class_<Value, std::shared_ptr<Value>>(m, "Value")
      .def(py::init<float>())
      .def_readwrite("data", &Value::data)
      .def_readwrite("grad", &Value::grad)
      .def("backward", &Value::backward)
      .def("__add__",
           [](const std::shared_ptr<Value> &a,
              const std::shared_ptr<Value> &b) { return Value::add(a, b); })
      .def("__mul__",
           [](const std::shared_ptr<Value> &a,
              const std::shared_ptr<Value> &b) { return Value::mul(a, b); });
}
