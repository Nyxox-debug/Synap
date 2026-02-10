#pragma once
#include "storage.h"
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
  Tensor(std::vector<size_t> shape, bool requires_grad = false);
  Tensor(StoragePtr storage, std::vector<size_t> shape,
         std::vector<size_t> stride, size_t offset, bool requires_grad);

  float *data();
  const float *data() const;
  const std::vector<size_t> &shape() const;

  Tensor clone() const;
  Tensor view(std::vector<size_t> new_shape) const;

  void zero_grad();
  void backward(std::shared_ptr<Tensor> grad_output);
  void set_values(const std::vector<float> &values);

  bool requires_grad;
  std::shared_ptr<Tensor> grad;

  friend std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor> &a,
                                     const std::shared_ptr<Tensor> &b);

  friend std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor> &a,
                                     const std::shared_ptr<Tensor> &b);

  friend std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor> &x);

private:
  StoragePtr storage_;
  std::vector<size_t> shape_;
  std::vector<size_t> stride_;
  size_t offset_;
  std::vector<std::shared_ptr<Tensor>> parents_;
  std::function<void()> backward_fn_;

private:
  static void build_topo(const std::shared_ptr<Tensor> &t,
                         std::unordered_set<Tensor *> &visited,
                         std::vector<std::shared_ptr<Tensor>> &topo);
};
