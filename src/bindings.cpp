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
      .def("view", &Tensor::view, py::arg("new_shape"))
      .def("set_values", &Tensor::set_values)
      .def("zero_grad", &Tensor::zero_grad)
      .def("backward",
           [](std::shared_ptr<Tensor> self) { self->backward(nullptr); })
      .def("backward",
           [](std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> grad) {
             self->backward(grad);
           })
      // .def("backward", [](Tensor &self) { self.backward(nullptr); })
      // .def("backward",
      //      [](Tensor &self, std::shared_ptr<Tensor> grad) {
      //        self.backward(grad);
      //      })
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

      // Add, Sub,  Mul and Div
      .def_static("add",
                  [](const std::shared_ptr<Tensor> &a,
                     const std::shared_ptr<Tensor> &b) { return add(a, b); })
      .def_static("sub",
                  [](const std::shared_ptr<Tensor> &a,
                     const std::shared_ptr<Tensor> &b) { return sub(a, b); })
      .def_static("mul",
                  [](const std::shared_ptr<Tensor> &a,
                     const std::shared_ptr<Tensor> &b) { return mul(a, b); })
      .def_static("div",
                  [](const std::shared_ptr<Tensor> &a,
                     const std::shared_ptr<Tensor> &b) { return div(a, b); })
      .def_static("mean",
                  [](const std::shared_ptr<Tensor> &a) { return mean(a); })
      .def_static("sigmoid",
                  [](const std::shared_ptr<Tensor> &a) { return sigmoid(a); })
      .def_static("tanh",
                  [](const std::shared_ptr<Tensor> &a) { return tanh(a); })
      .def_static("relu",
                  [](const std::shared_ptr<Tensor> &a) { return relu(a); })
      .def_static("mse",
                  [](const std::shared_ptr<Tensor> &pred,
                     const std::shared_ptr<Tensor> &target) {
                    return mse(pred, target);
                  })
      .def("backward", [](Tensor &self) { self.backward(nullptr); })
      .def_static("softmax_cross_entropy",
                  [](const std::shared_ptr<Tensor> &logits,
                     const std::shared_ptr<Tensor> &targets) {
                    return softmax_cross_entropy(logits, targets);
                  })

      .def_static("concat",
                  [](const std::vector<std::shared_ptr<Tensor>> &tensors) {
                    return concat(tensors);
                  })

      // Linear Algebra Ops
      .def_static("transpose",
                  [](const std::shared_ptr<Tensor> &a) { return transpose(a); })
      .def_static("matmul",
                  [](const std::shared_ptr<Tensor> &a,
                     const std::shared_ptr<Tensor> &b) { return matmul(a, b); })
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
