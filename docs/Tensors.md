# Tensors

An explanation of how `Tensor` works internally in Synap — storage, strides, views, and the computation graph.

---

## Overview

A `Tensor` in Synap is a view into a contiguous block of `float` memory. Multiple tensors can share the same underlying `Storage`, which is what makes operations like `view()` free — no data is copied.

```
┌─────────────────────────────────┐
│  Storage  [ f0, f1, f2, f3, … ] │  ← shared_ptr<Storage>
└─────────────────────────────────┘
         ▲             ▲
   Tensor A         Tensor B (view)
   shape: [2,3]     shape: [3,2]
   stride: [3,1]    stride: [2,1]
   offset: 0        offset: 0
```

---

## Storage

`Storage` is a plain struct holding a heap-allocated `float` array and its size:

```cpp
// storage.h
struct Storage {
    float* data;
    size_t size;
    explicit Storage(size_t n);
    ~Storage();
};
using StoragePtr = std::shared_ptr<Storage>;
```

Ownership is managed with `shared_ptr`. As long as any `Tensor` holds a reference to the storage, the memory stays alive — even after the original tensor goes out of scope.

---

## Shape, Stride, and Offset

Every `Tensor` carries three pieces of layout metadata:

| Field | Type | Meaning |
|---|---|---|
| `shape_` | `vector<size_t>` | Size of each dimension |
| `stride_` | `vector<size_t>` | Step (in elements) to advance one position in each dimension |
| `offset_` | `size_t` | Starting element index within storage |

For a newly allocated `[M, N]` tensor, the default strides are row-major (C order): `stride_ = [N, 1]`. The element at logical position `(i, j)` lives at `storage[offset_ + i * N + j]`.

A `view()` reuses the same storage but assigns new shape and stride values — making reshape a zero-copy operation.

---

## Constructors

### Allocating constructor

```cpp
Tensor(std::vector<size_t> shape, bool requires_grad = false);
```

Allocates fresh storage of `numel(shape)` floats. Computes row-major strides automatically. If `requires_grad` is true, also allocates a zero-filled `grad` tensor of the same shape.

### View constructor

```cpp
Tensor(StoragePtr storage,
       std::vector<size_t> shape,
       std::vector<size_t> stride,
       size_t offset,
       bool requires_grad);
```

Used internally by `view()` and `clone()`. Does not allocate new storage.

---

## Key Methods

### `data()`

Returns a raw pointer to the first element of this tensor's logical data:

```cpp
float* data() { return storage_->data + offset_; }
```

All ops read and write through `data()`. For contiguous row-major tensors, `data()[i]` is the `i`-th element in row-major order.

### `view(new_shape)`

Returns a new `Tensor` sharing the same storage, with a new shape and freshly computed row-major strides:

```cpp
auto t = synap.Tensor([6], requires_grad=True)
t.set_values([1, 2, 3, 4, 5, 6])

v = t.view([2, 3])  # no copy, same storage
```

Gradient flows back through `view()`: the backward function simply copies the upstream gradient element-wise back to the original shape.

### `clone()`

Allocates new storage and copies all values. The clone is independent — mutations to the clone do not affect the original.

### `set_values(values)`

Overwrites the tensor's data from a `vector<float>`. Raises if the size does not match `numel(shape)`.

### `zero_grad()`

Fills the `grad` tensor with zeros. Call this before each backward pass to clear accumulated gradients.

---

## Automatic Differentiation

### Computation Graph

During the forward pass, every operation that produces a tensor with `requires_grad = true` records:

- `parents_` — the input tensors it depends on
- `backward_fn_` — a closure that computes and accumulates `∂loss/∂input` for each parent

This builds a dynamic directed acyclic graph (DAG) of `Tensor` nodes.

### `backward()`

Calling `.backward()` on a scalar loss tensor triggers the reverse pass:

1. Sets `loss.grad = [1.0]` (or uses a provided upstream gradient)
2. Calls `build_topo()` — a depth-first traversal that produces a topological ordering of all ancestor nodes
3. Iterates in reverse topological order, calling each node's `backward_fn_()`

```python
loss = synap.Tensor.mse(pred, target)
loss.backward()
# Now pred.grad_values contains ∂loss/∂pred
```

### Gradient Accumulation

All backward functions use `+=` to write into `grad->data()`. This means:

- Gradients accumulate across calls — always call `zero_grad()` before the next backward pass
- A tensor used multiple times in a graph (fan-out) correctly receives the sum of all upstream gradients

---

## Supported Broadcasting

Synap implements a subset of NumPy-style broadcasting:

| Pattern | Example |
|---|---|
| Same shape | `[M, N] + [M, N]` |
| Scalar + tensor | `[1] + [M, N]` |
| Row broadcast | `[M, N] + [N]` |
| Column broadcast | `[M, N] + [M, 1]` |

Backward through a broadcast operation reduces the gradient back along the broadcast dimensions (summing over the axes that were expanded).

Unsupported broadcast patterns raise `std::runtime_error`.

---

## Operations and Their Gradients

| Op | Forward | Backward (chain rule) |
|---|---|---|
| `add(a, b)` | `a + b` | `∂a += grad`, `∂b += grad` |
| `sub(a, b)` | `a - b` | `∂a += grad`, `∂b -= grad` |
| `mul(a, b)` | `a * b` | `∂a += b * grad`, `∂b += a * grad` |
| `div(a, b)` | `a / b` | `∂a += grad / b`, `∂b -= grad * a / b²` |
| `matmul(a, b)` | `A @ B` | `∂A += grad @ Bᵀ`, `∂B += Aᵀ @ grad` |
| `transpose(x)` | `xᵀ` | `∂x += gradᵀ` |
| `relu(x)` | `max(0, x)` | `∂x += grad * (x > 0)` |
| `sigmoid(x)` | `σ(x)` | `∂x += σ(x)(1 − σ(x)) * grad` |
| `tanh(x)` | `tanh(x)` | `∂x += (1 − tanh²(x)) * grad` |
| `exp(x)` | `eˣ` | `∂x += eˣ * grad` |
| `sum(x)` | `Σxᵢ` | `∂xᵢ += grad[0]` for all `i` |
| `mean(x)` | `Σxᵢ / n` | `∂xᵢ += grad[0] / n` for all `i` |
| `softmax_cross_entropy` | numerically stable SCE | `∂logitsᵢ += (softmax(logits) − targets)ᵢ / n` |
| `concat(tensors)` | flat concatenation | gradient is split back to each input |
| `view(shape)` | reshape | element-wise copy back to original shape |

---

## Example: Manual Gradient Check

```python
import synap

x = synap.Tensor([1], requires_grad=True)
x.set_values([3.0])

# f(x) = x^2  →  f'(x) = 2x = 6
x2 = synap.Tensor.mul(x, x)
x2.backward()

print(x.grad_values)  # [6.0]
```
