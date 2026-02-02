#include "tensor.h"
#include <numeric>
#include <stdexcept>

static size_t numel(const std::vector<size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(),
                           size_t{1}, std::multiplies<>());
}

Tensor::Tensor(std::vector<size_t> shape, bool requires_grad)
    : shape_(shape), offset_(0), requires_grad(requires_grad) {

    size_t n = numel(shape);
    storage_ = std::make_shared<Storage>(n);

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

Tensor::Tensor(StoragePtr storage,
               std::vector<size_t> shape,
               std::vector<size_t> stride,
               size_t offset,
               bool requires_grad)
    : storage_(storage),
      shape_(shape),
      stride_(stride),
      offset_(offset),
      requires_grad(requires_grad) {}

float* Tensor::data() {
    return storage_->data + offset_;
}

const float* Tensor::data() const {
    return storage_->data + offset_;
}

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

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
    if (!grad) return;
    size_t n = numel(grad->shape_);
    std::fill(grad->data(), grad->data() + n, 0.0f);
}
