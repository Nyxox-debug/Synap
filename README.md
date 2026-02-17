# Synap

**A minimal deep learning framework written in C++ with Python bindings.**

Synap is a from-scratch autograd engine and tensor library — think micrograd, but with a typed C++ core and a clean Python API via pybind11. It supports forward computation, reverse-mode automatic differentiation, and a simple neural network module system for building and training models.


## Features

- **Tensor** — n-dimensional float array with shape, stride, and offset, backed by shared storage
- **Autograd** — reverse-mode autodiff via dynamic computation graph (topological sort + backward pass)
- **Operations** — `add`, `sub`, `mul`, `div`, `matmul`, `transpose`, `relu`, `sigmoid`, `tanh`, `exp`, `sum`, `mean`, `mse`, `softmax_cross_entropy`, `concat`, `view`, `clone`
- **Broadcasting** — scalar-tensor, row broadcast `[M,N] + [N]`, column broadcast `[M,N] + [M,1]`
- **Python bindings** — full pybind11 module (`synap`) with type stubs
- **Neural network module** — `Neuron`, `Layer`, `MLP` in pure Python using the `synap` tensor API

---

## Project Structure

```
Synap/
├── src/
│   ├── synap/
│   │   ├── tensor.h          # Tensor class declaration
│   │   ├── tensor.cpp        # Tensor ops + autodiff
│   │   ├── storage.h         # Shared float storage
│   │   ├── autodiff.h
│   │   └── autodiff.cpp
│   └── bindings.cpp          # pybind11 module definition
├── python/
│   ├── nn.py                 # Module, Neuron, Layer, MLP
│   └── test_grad_descent.py         # Gradient descent demo
│   └── test_backwardpass.py         # Backward pass demo
├── stubs/
│   └── synap.pyi             # Python type stubs
├── docs/
│   ├── setup.md              # Build + install instructions
│   └── Tensors.md            # Tensor internals reference
├── CMakeLists.txt
└── CMakePresets.json
```

---

## Quick Start

### Build

```bash
cmake --preset default
cmake --build build
```

See [`docs/setup.md`](docs/setup.md) for detailed environment setup, including Python version requirements and virtual environment configuration.

### Install Python module

After building, the `synap` module is available in the build output. Add it to your `PYTHONPATH` or install it directly.

---

## Usage

### Tensors

```python
import synap

# Create a 2x3 tensor
t = synap.Tensor([2, 3], requires_grad=True)
t.set_values([1, 2, 3, 4, 5, 6])

# Element-wise ops
a = synap.Tensor([3], requires_grad=True)
a.set_values([1.0, 2.0, 3.0])

b = synap.Tensor([3], requires_grad=True)
b.set_values([4.0, 5.0, 6.0])

c = synap.Tensor.add(a, b)
loss = synap.Tensor.sum(c)
loss.backward()

print(a.grad_values)  # [1.0, 1.0, 1.0]
```

### Neural Network

```python
import synap
import nn  # python/nn.py

# Input: shape [1, 4]
x = synap.Tensor([1, 4], requires_grad=False)
x.set_values([1.0, 2.0, 3.0, 4.0])

# Target
y = synap.Tensor([1, 1], requires_grad=False)
y.set_values([10.0])

# MLP: 4 inputs → hidden layer of 4 → 1 output
model = nn.MLP(4, [4, 1])

# Training loop
lr = 0.01
for _ in range(100):
    out = model(x)
    loss = synap.Tensor.mse(out, y)

    model.zero_grad()
    loss.backward()

    for param in model.parameters():
        vals = synap.tensor_data(param)
        grads = param.grad_values
        param.set_values([v - lr * g for v, g in zip(vals, grads)])
```

---

## Operations Reference

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
| `.view(new_shape)` | Reshape (must preserve element count) |
| `.clone()` | Deep copy |
| `.backward()` | Trigger reverse-mode autodiff |
| `.zero_grad()` | Zero out gradient buffer |

---

## Autodiff

Synap builds a dynamic computation graph during the forward pass. Each tensor records its parent tensors and a `backward_fn` closure. Calling `.backward()` on a scalar tensor performs a reverse topological traversal, accumulating gradients via the chain rule.

```
loss.backward()
# Internally:
# 1. Sets loss.grad = [1.0]
# 2. Builds topological order of the computation graph
# 3. Calls backward_fn() on each node in reverse order
```

Gradients accumulate with `+=`, so `zero_grad()` must be called before each backward pass.

---

## Docs

- [`docs/setup.md`](docs/setup.md) — Build environment, CMake configuration, pybind11 setup
- [`docs/Tensors.md`](docs/Tensors.md) — Tensor internals: storage, strides, offsets, views

---

## License

MIT
