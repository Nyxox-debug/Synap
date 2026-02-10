#include "tensor.h"
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

static size_t numel(const std::vector<size_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t{1},
                         std::multiplies<>());
}

Tensor::Tensor(std::vector<size_t> shape, bool requires_grad)
    : shape_(shape), offset_(0), requires_grad(requires_grad) {

  size_t n = numel(shape); // Total number of elements in Tensor
  storage_ = std::make_shared<Storage>(n);
  backward_fn_ = [] {};

  if (requires_grad) {
    grad = std::make_shared<Tensor>(shape, false);
  }

  stride_.resize(shape.size()); // NOTE: .resize ensures stride has one element
                                // per dimension
  size_t s = 1;
  for (int i = shape.size() - 1; i >= 0; --i) { // i represents a dimension
    stride_[i] = s;
    s *= shape[i];
  }

  if (requires_grad) {
    grad = std::make_shared<Tensor>(shape, false);
  }
}

// NOTE: View into existing Tensor memory
Tensor::Tensor(StoragePtr storage, std::vector<size_t> shape,
               std::vector<size_t> stride, size_t offset, bool requires_grad)
    : storage_(storage), shape_(shape), stride_(stride), offset_(offset),
      requires_grad(requires_grad) {
  if (requires_grad) {
    grad = std::make_shared<Tensor>(shape, false);
  }
}

float *Tensor::data() { return storage_->data + offset_; }

const float *Tensor::data() const { return storage_->data + offset_; }

const std::vector<size_t> &Tensor::shape() const { return shape_; }

Tensor Tensor::clone() const {
  Tensor out(shape_, requires_grad);
  size_t n = numel(shape_);
  std::copy(data(), data() + n, out.data());
  return out;
}

Tensor Tensor::view(std::vector<size_t> new_shape) const {
  if (numel(new_shape) != numel(shape_))
    throw std::runtime_error("View must preserve number of elements");

  return Tensor(storage_, new_shape, stride_, offset_, requires_grad);
}

void Tensor::zero_grad() {
  if (!grad)
    return;
  size_t n = numel(grad->shape_);
  std::fill(grad->data(), grad->data() + n, 0.0f);
}

void Tensor::build_topo(const std::shared_ptr<Tensor> &t,
                        std::unordered_set<Tensor *> &visited,
                        std::vector<std::shared_ptr<Tensor>> &topo) {

  if (visited.count(t.get()))
    return;
  visited.insert(t.get());

  for (auto &p : t->parents_) {
    build_topo(p, visited, topo);
  }

  topo.push_back(t);
}

void Tensor::backward(std::shared_ptr<Tensor> grad_output) {
  // Allocate grad if needed
  if (!grad) {
    grad = std::make_shared<Tensor>(shape_, false);
  }

  size_t n = numel(shape_);

  if (grad_output) {
    // Explicit upstream gradient
    std::copy(grad_output->data(), grad_output->data() + n, grad->data());
  } else {
    // Implicit upstream gradient (PyTorch-style)
    // Treat as: L = sum(this)
    std::fill(grad->data(), grad->data() + n, 1.0f);
  }

  // Build topological order
  std::vector<std::shared_ptr<Tensor>> topo;
  std::unordered_set<Tensor *> visited;
  build_topo(shared_from_this(), visited, topo);

  // Backpropagate
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    (*it)->backward_fn_();
  }
}

std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor> &a,
                            const std::shared_ptr<Tensor> &b) {
  if (a->shape_ != b->shape_)
    throw std::runtime_error("Shape mismatch in add");

  auto out =
      std::make_shared<Tensor>(a->shape_, a->requires_grad || b->requires_grad);

  size_t n = numel(a->shape_);
  for (size_t i = 0; i < n; ++i) {
    out->data()[i] = a->data()[i] * b->data()[i];
  }

  if (out->requires_grad) {
    out->parents_ = {a, b}; // NOTE: allowed because mul is a friend

    out->backward_fn_ = [out, a, b]() {
      size_t n = numel(out->shape_);
      for (size_t i = 0; i < n; ++i) {
        if (a->requires_grad)
          a->grad->data()[i] += b->data()[i] * out->grad->data()[i];
        if (a->requires_grad)
          b->grad->data()[i] += a->data()[i] * out->grad->data()[i];
      }
    };
  }

  return out;
};

std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor> &a,
                            const std::shared_ptr<Tensor> &b) {
  if (a->shape_ != b->shape_)
    throw std::runtime_error("Shape mismatch in add");

  auto out =
      std::make_shared<Tensor>(a->shape_, a->requires_grad || b->requires_grad);

  size_t n = numel(a->shape_);
  for (size_t i = 0; i < n; ++i) {
    out->data()[i] = a->data()[i] + b->data()[i];
  }

  if (out->requires_grad) {
    out->parents_ = {a, b}; // NOTE: allowed because add is a friend

    out->backward_fn_ = [out, a, b]() {
      size_t n = numel(out->shape_);
      for (size_t i = 0; i < n; ++i) {
        if (a->requires_grad)
          a->grad->data()[i] += out->grad->data()[i];
        if (a->requires_grad)
          b->grad->data()[i] += out->grad->data()[i];
      }
    };
  }

  return out;
}

std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor> &x) {
    auto out = std::make_shared<Tensor>(std::vector<size_t>{1}, x->requires_grad);

    size_t n = numel(x->shape_);
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i)
        s += x->data()[i];
    out->data()[0] = s;

    if (out->requires_grad) {
        out->parents_ = {x};
        out->backward_fn_ = [out, x, n]() {
            // grad of sum w.r.t each element is 1
            for (size_t i = 0; i < n; ++i)
                x->grad->data()[i] += out->grad->data()[0];
        };
    }

    return out;
}


void Tensor::set_values(const std::vector<float>& values) {
    size_t n = numel(shape_);
    if (values.size() != n)
        throw std::runtime_error("set_values: size mismatch");
    std::copy(values.begin(), values.end(), data());
}
