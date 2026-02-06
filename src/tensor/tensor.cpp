// FIX: 3 errs
#include "tensor.h"
#include <numeric>
#include <stdexcept>
#include <unordered_set>

static size_t numel(const std::vector<size_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t{1},
                         std::multiplies<>());
}

Tensor::Tensor(std::vector<size_t> shape, bool requires_grad)
    : shape_(shape), offset_(0), requires_grad(requires_grad) {

  size_t n = numel(shape);
  storage_ = std::make_shared<Storage>(n);
  backward_fn_ = [] {};

  stride_.resize(shape.size());
  size_t s = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    stride_[i] = s;
    s *= shape[i];
  }

  if (requires_grad) {
    grad = std::make_shared<Tensor>(shape, false);
  }
}

Tensor::Tensor(StoragePtr storage, std::vector<size_t> shape,
               std::vector<size_t> stride, size_t offset, bool requires_grad)
    : storage_(storage), shape_(shape), stride_(stride), offset_(offset),
      requires_grad(requires_grad) {}

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

void Tensor::backward() {
  if (numel(shape_) != 1)
    throw std::runtime_error("backward() only supported for scalar tensors");

  if (!grad) {
    grad = std::make_shared<Tensor>(shape_, false);
  }

  grad->data()[0] = 1.0f;

  std::vector<std::shared_ptr<Tensor>> topo;
  std::unordered_set<Tensor *> visited;

  build_topo(shared_from_this(), visited, topo);

  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    (*it)->backward_fn_();
  }
}

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
    out->parents_ = {a, b}; // allowed because add is a friend

    out->backward_fn_ = [out, a, b]() {
      size_t n = numel(out->shape_);
      for (size_t i = 0; i < n; ++i) {
        if (a->grad)
          a->grad->data()[i] += out->grad->data()[i];
        if (b->grad)
          b->grad->data()[i] += out->grad->data()[i];
      }
    };
  }

  return out;
}
