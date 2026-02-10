#include "autodiff.h"

Value::Value(float data)
    : data(data), grad(0.0f), backward_fn([] {}) {}

std::shared_ptr<Value> Value::add(
    const std::shared_ptr<Value>& a,
    const std::shared_ptr<Value>& b) {

  auto out = std::make_shared<Value>(a->data + b->data);
  out->parents = {a, b};

  out->backward_fn = [out, a, b]() {
    a->grad += out->grad;
    b->grad += out->grad;
  };

  return out;
}

std::shared_ptr<Value> Value::mul(
    const std::shared_ptr<Value>& a,
    const std::shared_ptr<Value>& b) {

  auto out = std::make_shared<Value>(a->data * b->data);
  out->parents = {a, b};

  out->backward_fn = [out, a, b]() {
    a->grad += b->data * out->grad;
    b->grad += a->data * out->grad;
  };

  return out;
}

static void build_topo(
    const std::shared_ptr<Value>& v,
    std::unordered_set<Value*>& visited,
    std::vector<std::shared_ptr<Value>>& topo) {

  if (visited.count(v.get())) return;
  visited.insert(v.get());

  for (auto& p : v->parents) {
    build_topo(p, visited, topo);
  }

  topo.push_back(v);
}

void Value::backward() {
  std::vector<std::shared_ptr<Value>> topo;
  std::unordered_set<Value*> visited;

  build_topo(shared_from_this(), visited, topo);

  grad = 1.0f;  // ∂output/∂output

  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    (*it)->backward_fn();
  }
}
