# Synap

A minimal deep learning framework written in C++ with Python bindings.

![Gradient Descent test](as/im.png)

Synap is a from-scratch autograd engine and tensor library — think micrograd, but with a typed C++ core and a clean Python API via pybind11.

```
Synap/
├── src/
│   ├── synap/
│   │   ├── tensor.h / tensor.cpp   # Tensor ops + autodiff
│   │   ├── storage.h               # Shared float storage
│   │   └── scalar-based.h / scalar-based.cpp
│   └── bindings.cpp                # pybind11 module
├── python/
│   ├── nn.py                       # Neuron, Layer, MLP
│   ├── test_grad_descent.py
│   └── test_backwardpass.py
├── stubs/synap.pyi
├── docs/
│   ├── setup.md
│   ├── Tensors.md
│   └── Operations.md
└── CMakeLists.txt
```

---

## Build

```bash
cmake --preset setup
cmake --build build
```

See [`docs/setup.md`](docs/setup.md) for environment setup, Python version requirements, and virtual environment configuration.

---

## Quick Start

### Gradient Descent Test

```python
import nn
import synap

def test_gradient_descent():
    """Simple gradient descent test - trains for a few iterations"""
    
    # Input
    # NOTE: I set requires_grad is false becuase it is the input for the neural network, which is the data
    x = synap.Tensor([1, 4], requires_grad=False)
    x.set_values([1, 2, 3, 4])
    
    # Target
    y = synap.Tensor([1, 1], requires_grad=False)
    y.set_values([10])
    
    # Model
    m = nn.MLP(4, [4, 1])
    
    print("₄" * 50)
    print("Training with Gradient Descent")
    print("₄" * 50)
    
    # Grab a weight to track
    w = m.layers[0].neurons[0].w
    print(f"Initial weight: {synap.tensor_data(w)}")
    
    # Training loop
    learning_rate = 0.1
    for iteration in range(200):
        # Forward
        out = m(x)
        
        # Loss (MSE)
        loss = synap.Tensor.mse(out, y)
        loss_val = synap.tensor_data(loss)[0]
        
        # Backward
        m.zero_grad()
        loss.backward()
        
        # Update parameters (gradient descent step) - This is an Explicitly written Gradient Descent 
        for param in m.parameters():
            param_values = synap.tensor_data(param)
            grad_values = param.grad_values
            
            # param = param - learning_rate * grad
            # NOTE: Move opposite the gradient.
            # The sign of grad encodes whether increasing this parameter raises or lowers the loss.
            updated = [p - learning_rate * g for p, g in zip(param_values, grad_values)]
            param.set_values(updated)
        
        # Print progress
        if iteration % 2 == 0:
            print(f"Epoch {iteration}: Loss = {loss_val:.4f}")
    
    print(f"\nFinal weight: {synap.tensor_data(w)}")
    print(f"Weight changed: {synap.tensor_data(w)[0] - synap.tensor_data(w)[0] != 0}")
    
    # Final forward pass
    final_out = m(x)
    final_loss = synap.Tensor.mse(final_out, y)
    print(f"Final loss: {synap.tensor_data(final_loss)[0]:.4f}")
    print(f"Final output: {synap.tensor_data(final_out)}")
    print(f"Target: {synap.tensor_data(y)}")

if __name__ == "__main__":
    test_gradient_descent()
```

### Tensors

```python
import synap

a = synap.Tensor([3], requires_grad=True)
a.set_values([1.0, 2.0, 3.0])

b = synap.Tensor([3], requires_grad=True)
b.set_values([4.0, 5.0, 6.0])

loss = synap.Tensor.sum(synap.Tensor.add(a, b))
loss.backward()

print(a.grad_values)  # [1.0, 1.0, 1.0]
```

### Training a Neural Network

```python
import synap
import nn  # python/nn.py

x = synap.Tensor([1, 4], requires_grad=False)
x.set_values([1.0, 2.0, 3.0, 4.0])

y = synap.Tensor([1, 1], requires_grad=False)
y.set_values([10.0])

model = nn.MLP(4, [4, 1])
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

## Docs

- [`docs/setup.md`](docs/setup.md) — Build environment and CMake configuration
- [`docs/Tensors.md`](docs/Tensors.md) — Tensor internals: storage, strides, views, and the computation graph
- [`docs/Operations.md`](docs/Operations.md) — Full operations reference and autodiff rules

---

## License

MIT
