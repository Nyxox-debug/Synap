#pragma once
#include "storage.h"
#include <memory>
#include <vector>

class Tensor {
public:
  Tensor(std::vector<size_t> shape, bool requires_grad = false);

  // view constructor, used to setup metadata for Linear memory
  Tensor(StoragePtr storage, std::vector<size_t> shape,
         std::vector<size_t> stride, size_t offset, bool requires_grad);

  float *data();
  const float *data() const;
  const std::vector<size_t> &shape() const;

  Tensor clone() const;
  Tensor view(std::vector<size_t> new_shape) const;

  void zero_grad();

  bool requires_grad;
  std::shared_ptr<Tensor> grad;

private:
  StoragePtr storage_;
  std::vector<size_t> shape_;
  std::vector<size_t> stride_;
  size_t offset_;
};
