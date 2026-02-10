#include "synap/tensor.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(synap, m) {
  m.doc() = "Synap — a mini deep learning framework with Tensors and autodiff.";
  py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
      .def(py::init<std::vector<size_t>, bool>(), py::arg("shape"),
           py::arg("requires_grad") = false)
      .def("shape", &Tensor::shape)
      .def("clone", &Tensor::clone)
      .def("view", &Tensor::view)
      .def("set_values", &Tensor::set_values)
      .def("zero_grad", &Tensor::zero_grad)
      .def("backward", [](Tensor &self) { self.backward(nullptr); })
      .def("backward",
           [](Tensor &self, std::shared_ptr<Tensor> grad) {
             self.backward(grad);
           })
      .def_readonly("requires_grad", &Tensor::requires_grad)
      .def_readwrite("grad", &Tensor::grad)
      .def_property_readonly("grad_values",
                             [](std::shared_ptr<Tensor> t) {
                               if (!t->grad)
                                 return std::vector<float>{};
                               size_t n = 1;
                               for (auto s : t->grad->shape())
                                 n *= s;
                               return std::vector<float>(t->grad->data(),
                                                         t->grad->data() + n);
                             })

      // Sum and Mul
      .def_static("add",
                  [](const std::shared_ptr<Tensor> &a,
                     const std::shared_ptr<Tensor> &b) { return add(a, b); })
      .def_static("mul",
                  [](const std::shared_ptr<Tensor> &a,
                     const std::shared_ptr<Tensor> &b) { return mul(a, b); })
      // Scalar Sink
      .def_static("sum",
                  [](const std::shared_ptr<Tensor> &a) { return sum(a); });

  // Exposing Data for python with a python List
  m.def("tensor_data", [](std::shared_ptr<Tensor> t) {
    size_t n = 1;
    for (auto s : t->shape())
      n *= s;
    std::vector<float> vec(t->data(), t->data() + n);
    return vec;
  });
}
