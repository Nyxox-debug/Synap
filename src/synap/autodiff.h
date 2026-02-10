#pragma once
#include <functional>
#include <memory>
#include <vector>
#include <unordered_set>

struct Value : std::enable_shared_from_this<Value> {
  float data;
  float grad;

  std::vector<std::shared_ptr<Value>> parents;
  std::function<void()> backward_fn;

  explicit Value(float data);

  static std::shared_ptr<Value> add(
      const std::shared_ptr<Value>& a,
      const std::shared_ptr<Value>& b);

  static std::shared_ptr<Value> mul(
      const std::shared_ptr<Value>& a,
      const std::shared_ptr<Value>& b);

  void backward();
};
