#include "tensor.h"
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

static size_t numel(const std::vector<size_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t{1},
                         std::multiplies<>());
}

static bool is_scalar(const std::shared_ptr<Tensor> &t) {
  return numel(t->shape_) == 1;
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
  backward_fn_ = [] {};
  if (requires_grad) {
    grad = std::make_shared<Tensor>(shape, false);
  }
}

float *Tensor::data() { return storage_->data + offset_; }

const float *Tensor::data() const { return storage_->data + offset_; }

const std::vector<size_t> &Tensor::shape() const { return shape_; }

// Tensor Tensor::clone() const {
//   Tensor out(shape_, requires_grad);
//   size_t n = numel(shape_);
//   std::copy(data(), data() + n, out.data());
//   return out;
// }

// Tensor Tensor::view(std::vector<size_t> new_shape) const {
//   if (numel(new_shape) != numel(shape_))
//     throw std::runtime_error("View must preserve number of elements");
//
//   return Tensor(storage_, new_shape, stride_, offset_, requires_grad);
// }

std::shared_ptr<Tensor> Tensor::clone() const {
    auto out = std::make_shared<Tensor>(shape_, requires_grad);
    size_t n = numel(shape_);
    std::copy(data(), data() + n, out->data());
    return out;
}

std::shared_ptr<Tensor> Tensor::view(const std::vector<size_t>& new_shape) const {
    if (numel(new_shape) != numel(shape_))
        throw std::runtime_error("View must preserve number of elements");

    std::vector<size_t> new_stride(new_shape.size());
    size_t stride = 1;
    for (int i = new_shape.size() - 1; i >= 0; --i) {
        new_stride[i] = stride;
        stride *= new_shape[i];
    }

    return std::make_shared<Tensor>(
        storage_,
        new_shape,
        new_stride,
        offset_,
        requires_grad
    );
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
  // Same shape
  if (a->shape_ == b->shape_) {
    auto out = std::make_shared<Tensor>(a->shape_,
                                        a->requires_grad || b->requires_grad);

    size_t n = numel(a->shape_);
    for (size_t i = 0; i < n; ++i)
      out->data()[i] = a->data()[i] * b->data()[i];

    if (out->requires_grad) {
      out->parents_ = {a, b};
      out->backward_fn_ = [out, a, b, n]() {
        for (size_t i = 0; i < n; ++i) {
          if (a->requires_grad)
            a->grad->data()[i] += b->data()[i] * out->grad->data()[i];
          if (b->requires_grad)
            b->grad->data()[i] += a->data()[i] * out->grad->data()[i];
        }
      };
    }
    return out;
  }

  // Scalar + tensor
  if (is_scalar(a)) {
    auto out = std::make_shared<Tensor>(b->shape_,
                                        a->requires_grad || b->requires_grad);
    float av = a->data()[0];
    size_t n = numel(b->shape_);

    for (size_t i = 0; i < n; ++i)
      out->data()[i] = av * b->data()[i];

    if (out->requires_grad) {
      out->parents_ = {a, b};
      out->backward_fn_ = [out, a, b, n, av]() {
        float grad_sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
          if (b->requires_grad)
            b->grad->data()[i] += av * out->grad->data()[i];
          grad_sum += b->data()[i] * out->grad->data()[i];
        }
        if (a->requires_grad)
          a->grad->data()[0] += grad_sum;
      };
    }
    return out;
  }

  if (is_scalar(b))
    return mul(b, a);

  throw std::runtime_error("Unsupported broadcast in mul");
}

std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor> &a,
                            const std::shared_ptr<Tensor> &b) {

  // Case 1: same shape → existing behavior
  if (a->shape_ == b->shape_) {
    auto out = std::make_shared<Tensor>(a->shape_,
                                        a->requires_grad || b->requires_grad);

    size_t n = numel(a->shape_);
    for (size_t i = 0; i < n; ++i)
      out->data()[i] = a->data()[i] + b->data()[i];

    if (out->requires_grad) {
      out->parents_ = {a, b};
      out->backward_fn_ = [out, a, b, n]() {
        for (size_t i = 0; i < n; ++i) {
          if (a->requires_grad)
            a->grad->data()[i] += out->grad->data()[i];
          if (b->requires_grad)
            b->grad->data()[i] += out->grad->data()[i];
        }
      };
    }
    return out;
  }

  // Case 2: scalar + tensor
  if (is_scalar(a)) {
    auto out = std::make_shared<Tensor>(b->shape_,
                                        a->requires_grad || b->requires_grad);

    size_t n = numel(b->shape_);
    float av = a->data()[0];

    for (size_t i = 0; i < n; ++i)
      out->data()[i] = av + b->data()[i];

    if (out->requires_grad) {
      out->parents_ = {a, b};
      out->backward_fn_ = [out, a, b, n]() {
        float grad_sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
          if (b->requires_grad)
            b->grad->data()[i] += out->grad->data()[i];
          grad_sum += out->grad->data()[i];
        }
        if (a->requires_grad)
          a->grad->data()[0] += grad_sum;
      };
    }
    return out;
  }

  // Case 3: tensor + scalar
  if (is_scalar(b)) {
    return add(b, a); // reuse logic
  }

  // Case 4: row broadcast  [M,N] + [N]
  if (a->shape_.size() == 2 && b->shape_.size() == 1 &&
      a->shape_[1] == b->shape_[0]) {

    size_t M = a->shape_[0];
    size_t N = a->shape_[1];

    auto out = std::make_shared<Tensor>(a->shape_,
                                        a->requires_grad || b->requires_grad);

    // Forward
    for (size_t i = 0; i < M; ++i)
      for (size_t j = 0; j < N; ++j)
        out->data()[i * N + j] = a->data()[i * N + j] + b->data()[j];

    if (out->requires_grad) {
      out->parents_ = {a, b};
      out->backward_fn_ = [out, a, b, M, N]() {
        for (size_t i = 0; i < M; ++i) {
          for (size_t j = 0; j < N; ++j) {

            float g = out->grad->data()[i * N + j];

            if (a->requires_grad)
              a->grad->data()[i * N + j] += g;

            if (b->requires_grad)
              b->grad->data()[j] += g; // reduce over rows
          }
        }
      };
    }

    return out;
  }

  // Case 5: column broadcast  [M,N] + [M,1]
  if (a->shape_.size() == 2 && b->shape_.size() == 2 &&
      b->shape_[0] == a->shape_[0] && b->shape_[1] == 1) {

    size_t M = a->shape_[0];
    size_t N = a->shape_[1];

    auto out = std::make_shared<Tensor>(a->shape_,
                                        a->requires_grad || b->requires_grad);

    // Forward
    for (size_t i = 0; i < M; ++i)
      for (size_t j = 0; j < N; ++j)
        out->data()[i * N + j] = a->data()[i * N + j] + b->data()[i];

    if (out->requires_grad) {
      out->parents_ = {a, b};
      out->backward_fn_ = [out, a, b, M, N]() {
        for (size_t i = 0; i < M; ++i) {
          for (size_t j = 0; j < N; ++j) {

            float g = out->grad->data()[i * N + j];

            if (a->requires_grad)
              a->grad->data()[i * N + j] += g;

            if (b->requires_grad)
              b->grad->data()[i] += g; // reduce over columns
          }
        }
      };
    }

    return out;
  }

  throw std::runtime_error("Unsupported broadcast in add");
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

std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor> &a,
                            const std::shared_ptr<Tensor> &b) {
  // Same shape
  if (a->shape_ == b->shape_) {
    auto out = std::make_shared<Tensor>(a->shape_,
                                        a->requires_grad || b->requires_grad);
    size_t n = numel(a->shape_);
    for (size_t i = 0; i < n; ++i)
      out->data()[i] = a->data()[i] - b->data()[i];

    if (out->requires_grad) {
      out->parents_ = {a, b};
      out->backward_fn_ = [out, a, b, n]() {
        for (size_t i = 0; i < n; ++i) {
          if (a->requires_grad)
            a->grad->data()[i] += out->grad->data()[i];
          if (b->requires_grad)
            b->grad->data()[i] -= out->grad->data()[i];
        }
      };
    }
    return out;
  }

  // Scalar + tensor
  if (is_scalar(a)) {
    auto out = std::make_shared<Tensor>(b->shape_,
                                        a->requires_grad || b->requires_grad);
    float av = a->data()[0];
    size_t n = numel(b->shape_);
    for (size_t i = 0; i < n; ++i)
      out->data()[i] = av - b->data()[i];

    if (out->requires_grad) {
      out->parents_ = {a, b};
      out->backward_fn_ = [out, a, b, n]() {
        float grad_sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
          if (b->requires_grad)
            b->grad->data()[i] -= out->grad->data()[i];
          grad_sum += out->grad->data()[i];
        }
        if (a->requires_grad)
          a->grad->data()[0] += grad_sum;
      };
    }
    return out;
  }

  if (is_scalar(b)) {
    // a - scalar = -(scalar - a) ?
    auto out = std::make_shared<Tensor>(a->shape_,
                                        a->requires_grad || b->requires_grad);
    float bv = b->data()[0];
    size_t n = numel(a->shape_);
    for (size_t i = 0; i < n; ++i)
      out->data()[i] = a->data()[i] - bv;

    if (out->requires_grad) {
      out->parents_ = {a, b};
      out->backward_fn_ = [out, a, b, n, bv]() {
        float grad_sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
          if (a->requires_grad)
            a->grad->data()[i] += out->grad->data()[i];
          grad_sum -= out->grad->data()[i];
        }
        if (b->requires_grad)
          b->grad->data()[0] += grad_sum;
      };
    }
    return out;
  }

  throw std::runtime_error("Unsupported broadcast in sub");
}

std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor> &a,
                            const std::shared_ptr<Tensor> &b) {
  // Same shape
  if (a->shape_ == b->shape_) {
    auto out = std::make_shared<Tensor>(a->shape_,
                                        a->requires_grad || b->requires_grad);
    size_t n = numel(a->shape_);
    for (size_t i = 0; i < n; ++i)
      out->data()[i] = a->data()[i] / b->data()[i];

    if (out->requires_grad) {
      out->parents_ = {a, b};
      out->backward_fn_ = [out, a, b, n]() {
        for (size_t i = 0; i < n; ++i) {
          if (a->requires_grad)
            a->grad->data()[i] += out->grad->data()[i] / b->data()[i];
          if (b->requires_grad)
            b->grad->data()[i] -= out->grad->data()[i] * a->data()[i] /
                                  (b->data()[i] * b->data()[i]);
        }
      };
    }
    return out;
  }

  // Scalar + tensor
  if (is_scalar(a)) {
    auto out = std::make_shared<Tensor>(b->shape_,
                                        a->requires_grad || b->requires_grad);
    float av = a->data()[0];
    size_t n = numel(b->shape_);
    for (size_t i = 0; i < n; ++i)
      out->data()[i] = av / b->data()[i];

    if (out->requires_grad) {
      out->parents_ = {a, b};
      out->backward_fn_ = [out, a, b, n]() {
        float grad_sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
          if (b->requires_grad)
            b->grad->data()[i] -= out->grad->data()[i] * a->data()[0] /
                                  (b->data()[i] * b->data()[i]);
          grad_sum += out->grad->data()[i] / b->data()[i];
        }
        if (a->requires_grad)
          a->grad->data()[0] += grad_sum;
      };
    }
    return out;
  }

  if (is_scalar(b)) {
    auto out = std::make_shared<Tensor>(a->shape_,
                                        a->requires_grad || b->requires_grad);
    float bv = b->data()[0];
    size_t n = numel(a->shape_);
    for (size_t i = 0; i < n; ++i)
      out->data()[i] = a->data()[i] / bv;

    if (out->requires_grad) {
      out->parents_ = {a, b};
      out->backward_fn_ = [out, a, b, n, bv]() {
        for (size_t i = 0; i < n; ++i) {
          if (a->requires_grad)
            a->grad->data()[i] += out->grad->data()[i] / bv;
        }
        if (b->requires_grad) {
          float grad_sum = 0.0f;
          for (size_t i = 0; i < n; ++i)
            grad_sum -= out->grad->data()[i] * a->data()[i] / (bv * bv);
          b->grad->data()[0] += grad_sum;
        }
      };
    }
    return out;
  }

  throw std::runtime_error("Unsupported broadcast in div");
}

void Tensor::set_values(const std::vector<float> &values) {
  size_t n = numel(shape_);
  if (values.size() != n)
    throw std::runtime_error("set_values: size mismatch");
  std::copy(values.begin(), values.end(), data());
}

// std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor> &x) {
//   auto s = sum(x);
//   s->data()[0] /= numel(x->shape_);
//   return s;
// }

std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor> &x) {
  auto s = sum(x);

  float inv_n = 1.0f / numel(x->shape_);
  auto scale = std::make_shared<Tensor>(std::vector<size_t>{1}, false);

  scale->data()[0] = inv_n;

  return mul(s, scale);
}

std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor> &x) {
  if (x->shape_.size() != 2)
    throw std::runtime_error("transpose only supports 2D tensors");

  auto out = std::make_shared<Tensor>(
      std::vector<size_t>{x->shape_[1], x->shape_[0]}, x->requires_grad);

  for (size_t i = 0; i < x->shape_[0]; ++i) {
    for (size_t j = 0; j < x->shape_[1]; ++j) {
      out->data()[j * x->shape_[0] + i] = x->data()[i * x->shape_[1] + j];
    }
  }

  if (out->requires_grad) {
    out->parents_ = {x};
    out->backward_fn_ = [out, x]() {
      // grad of transpose is just transpose of grad
      for (size_t i = 0; i < x->shape_[0]; ++i)
        for (size_t j = 0; j < x->shape_[1]; ++j)
          x->grad->data()[i * x->shape_[1] + j] +=
              out->grad->data()[j * x->shape_[0] + i];
    };
  }

  return out;
}

std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor> &a,
                               const std::shared_ptr<Tensor> &b) {
  if (a->shape_.size() != 2 || b->shape_.size() != 2)
    throw std::runtime_error("matmul supports only 2D tensors");
  if (a->shape_[1] != b->shape_[0])
    throw std::runtime_error("matmul shape mismatch");

  size_t M = a->shape_[0];
  size_t K = a->shape_[1];
  size_t N = b->shape_[1];

  auto out = std::make_shared<Tensor>(std::vector<size_t>{M, N},
                                      a->requires_grad || b->requires_grad);

  for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k)
        sum += a->data()[i * K + k] * b->data()[k * N + j];
      out->data()[i * N + j] = sum;
    }

  if (out->requires_grad) {
    out->parents_ = {a, b};
    out->backward_fn_ = [out, a, b, M, K, N]() {
      // d(out)/da = grad_out * B^T
      // d(out)/db = A^T * grad_out
      for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < K; ++j)
          if (a->requires_grad)
            for (size_t n = 0; n < N; ++n)
              a->grad->data()[i * K + j] +=
                  out->grad->data()[i * N + n] * b->data()[j * N + n];

      for (size_t i = 0; i < K; ++i)
        for (size_t j = 0; j < N; ++j)
          if (b->requires_grad)
            for (size_t m = 0; m < M; ++m)
              b->grad->data()[i * N + j] +=
                  a->data()[m * K + i] * out->grad->data()[m * N + j];
    };
  }

  return out;
}

std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor> &x) {
  auto out = std::make_shared<Tensor>(x->shape_, x->requires_grad);
  size_t n = numel(x->shape_);

  for (size_t i = 0; i < n; ++i)
    out->data()[i] = std::max(0.0f, x->data()[i]);

  if (out->requires_grad) {
    out->parents_ = {x};
    out->backward_fn_ = [out, x, n]() {
      for (size_t i = 0; i < n; ++i)
        if (x->requires_grad)
          x->grad->data()[i] +=
              (x->data()[i] > 0 ? 1.0f : 0.0f) * out->grad->data()[i];
    };
  }

  return out;
}

std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor> &x) {
  auto out = std::make_shared<Tensor>(x->shape_, x->requires_grad);
  size_t n = numel(x->shape_);

  for (size_t i = 0; i < n; ++i)
    out->data()[i] = 1.0f / (1.0f + std::exp(-x->data()[i]));

  if (out->requires_grad) {
    out->parents_ = {x};
    out->backward_fn_ = [out, x, n]() {
      for (size_t i = 0; i < n; ++i)
        if (x->requires_grad)
          x->grad->data()[i] +=
              out->data()[i] * (1.0f - out->data()[i]) * out->grad->data()[i];
    };
  }

  return out;
}

std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor> &x) {
  auto out = std::make_shared<Tensor>(x->shape_, x->requires_grad);
  size_t n = numel(x->shape_);

  for (size_t i = 0; i < n; ++i)
    out->data()[i] = std::tanh(x->data()[i]);

  if (out->requires_grad) {
    out->parents_ = {x};
    out->backward_fn_ = [out, x, n]() {
      for (size_t i = 0; i < n; ++i)
        if (x->requires_grad)
          x->grad->data()[i] +=
              (1.0f - out->data()[i] * out->data()[i]) * out->grad->data()[i];
    };
  }

  return out;
}

std::shared_ptr<Tensor> mse(const std::shared_ptr<Tensor> &pred,
                            const std::shared_ptr<Tensor> &target) {
  auto diff = sub(pred, target);
  auto sq = mul(diff, diff);
  return mean(sq);
}

std::shared_ptr<Tensor>
softmax_cross_entropy(const std::shared_ptr<Tensor> &logits,
                      const std::shared_ptr<Tensor> &targets) {
  size_t batch = logits->shape()[0];
  size_t classes = logits->shape()[1];

  // Forward (same as your code)
  std::vector<float> max_vals(batch, -1e9);
  for (size_t i = 0; i < batch; ++i)
    for (size_t j = 0; j < classes; ++j)
      if (logits->data()[i * classes + j] > max_vals[i])
        max_vals[i] = logits->data()[i * classes + j];

  auto logits_shifted =
      std::make_shared<Tensor>(logits->shape(), logits->requires_grad);
  size_t n = batch * classes;
  for (size_t i = 0; i < batch; ++i)
    for (size_t j = 0; j < classes; ++j)
      logits_shifted->data()[i * classes + j] =
          logits->data()[i * classes + j] - max_vals[i];

  auto exp_logits =
      std::make_shared<Tensor>(logits->shape(), logits->requires_grad);
  for (size_t i = 0; i < n; ++i)
    exp_logits->data()[i] = std::exp(logits_shifted->data()[i]);

  std::vector<float> row_sums(batch, 0.0f);
  for (size_t i = 0; i < batch; ++i)
    for (size_t j = 0; j < classes; ++j)
      row_sums[i] += exp_logits->data()[i * classes + j];

  auto probs = std::make_shared<Tensor>(logits->shape(), logits->requires_grad);
  for (size_t i = 0; i < batch; ++i)
    for (size_t j = 0; j < classes; ++j)
      probs->data()[i * classes + j] =
          exp_logits->data()[i * classes + j] / row_sums[i];

  auto mul_targets =
      std::make_shared<Tensor>(logits->shape(), logits->requires_grad);
  for (size_t i = 0; i < n; ++i)
    mul_targets->data()[i] = -targets->data()[i] * std::log(probs->data()[i]);

  auto out = mean(mul_targets);

  // Backward hook
  if (out->requires_grad) {
    out->parents_ = {logits};
    out->backward_fn_ = [logits, probs, targets, n]() {
      for (size_t i = 0; i < n; ++i)
        if (logits->requires_grad)
          logits->grad->data()[i] +=
              (probs->data()[i] - targets->data()[i]) / n;
    };
  }

  return out;
}

std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor> &x) {
  auto out = std::make_shared<Tensor>(x->shape_, x->requires_grad);
  size_t n = numel(x->shape_);
  for (size_t i = 0; i < n; ++i)
    out->data()[i] = std::exp(x->data()[i]);

  if (out->requires_grad) {
    out->parents_ = {x};
    out->backward_fn_ = [out, x, n]() {
      for (size_t i = 0; i < n; ++i)
        x->grad->data()[i] += out->data()[i] * out->grad->data()[i];
    };
  }
  return out;
}

// TODO: Concat
