# Operations Reference

A complete reference for all operations available in Synap, including their autodiff rules.

---

## Operations

| Operation | Description |
|---|---|
| `Tensor.add(a, b)` | Element-wise addition (with broadcasting) |
| `Tensor.sub(a, b)` | Element-wise subtraction |
| `Tensor.mul(a, b)` | Element-wise multiplication |
| `Tensor.div(a, b)` | Element-wise division |
| `Tensor.matmul(a, b)` | Matrix multiplication (2D only) |
| `Tensor.transpose(a)` | Transpose a 2D tensor |
| `Tensor.relu(a)` | ReLU activation |
| `Tensor.sigmoid(a)` | Sigmoid activation |
| `Tensor.tanh(a)` | Tanh activation |
| `Tensor.exp(a)` | Element-wise exponential |
| `Tensor.sum(a)` | Sum all elements → scalar |
| `Tensor.mean(a)` | Mean of all elements → scalar |
| `Tensor.mse(pred, target)` | Mean squared error loss |
| `Tensor.softmax_cross_entropy(logits, targets)` | Numerically stable softmax + cross-entropy |
| `Tensor.concat(tensors)` | Concatenate a list of tensors into a flat 1D tensor |
| `.view(new_shape)` | Reshape without copying (element count must be preserved) |
| `.clone()` | Deep copy with independent storage |
| `.backward()` | Trigger reverse-mode autodiff from a scalar |
| `.zero_grad()` | Zero out gradient buffer before a new backward pass |

---

## Autodiff Rules

Synap builds a dynamic computation graph during the forward pass. Each tensor with `requires_grad=True` records its parent tensors and a `backward_fn` closure. Calling `.backward()` on a scalar performs a reverse topological traversal, accumulating gradients via the chain rule.

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
| `concat(tensors)` | flat concatenation | gradient is split back to each input slice |
| `view(shape)` | reshape (zero-copy) | element-wise copy back to original shape |

---

## Broadcasting

Synap implements a subset of NumPy-style broadcasting:

| Pattern | Example |
|---|---|
| Same shape | `[M, N] + [M, N]` |
| Scalar + tensor | `[1] + [M, N]` |
| Row broadcast | `[M, N] + [N]` |
| Column broadcast | `[M, N] + [M, 1]` |

Backward through a broadcast reduces the upstream gradient along the broadcast dimensions (summing over expanded axes). Unsupported patterns raise `std::runtime_error`.

---

## Example: Manual Gradient Check

```python
import synap

x = synap.Tensor([1], requires_grad=True)
x.set_values([3.0])

# f(x) = x²  →  f'(x) = 2x = 6
x2 = synap.Tensor.mul(x, x)
x2.backward()

print(x.grad_values)  # [6.0]
```

---

## Gradient Accumulation

All backward functions write to `grad->data()` with `+=`, so gradients accumulate across calls. Always call `zero_grad()` before each backward pass to reset accumulated gradients. Tensors used multiple times in a graph (fan-out) correctly receive the sum of all upstream gradients.
