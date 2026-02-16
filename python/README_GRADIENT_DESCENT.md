# Gradient Descent Test for Synap Framework

## Overview
This test suite verifies that your autodiff implementation works correctly by performing actual gradient descent on neural networks. It includes three tests:

1. **Simple Linear Regression** - Fit y = 3*x + 2
2. **Multi-variable Regression** - Fit f(x) = 2*x1 + 3*x2 - 1  
3. **Batch Gradient Descent** - Train on XOR-like problem

## Files Included

- `test_gradient_descent.py` - Main test suite with gradient descent
- `nn.py` - Neural network module (Module, Neuron, Layer, MLP classes)

## How to Run

<!-- TODO: Setting up synap or publising to pyip -->

### Prerequisites
You need to have [Compiled](../docs/setup.md) the `synap` C++ extension or have it installed with pip.


### Running the Tests

```bash
# Run the gradient descent tests
python test_gradient_descent.py
```

## What the Tests Do

### Test 1: Simple Linear Regression
- Learns a simple linear function: y = 3*x + 2
- Verifies that the learned weight ≈ 3.0 and bias ≈ 2.0
- Demonstrates basic gradient descent on 1D input → 1D output

### Test 2: Multi-variable Regression  
- Learns f(x) = 2*x1 + 3*x2 - 1 with noise
- Uses a 2 → 4 → 1 neural network architecture
- Runs for 100 epochs with learning rate 0.01
- Verifies loss decreases significantly (should see >80% reduction)
- Tests the trained model on unseen examples

### Test 3: Batch Gradient Descent
- Tackles the XOR problem (requires nonlinearity)
- Uses batch updates (accumulates gradients before updating)
- 2 → 4 → 1 architecture with ReLU activation
- 200 epochs to learn the non-linear decision boundary

## Expected Output

You should see output like:

```
============================================================
Simple Linear Regression Test
============================================================
Target function: y = 3*x + 2

Training for 50 epochs...

Epoch  0: Loss = 12.456789
Epoch 10: Loss = 2.345678
Epoch 20: Loss = 0.456789
Epoch 30: Loss = 0.089012
Epoch 40: Loss = 0.017234

Learned parameters:
  Weight: 2.9876 (expected ~3.0)
  Bias:   2.0123 (expected ~2.0)

============================================================
Training with Gradient Descent
============================================================
Dataset size: 20
Learning rate: 0.01
Epochs: 100

Epoch   0: Loss = 45.678901
Epoch  10: Loss = 12.345678
...
Epoch  90: Loss = 0.234567
Epoch  99: Loss = 0.123456

============================================================
Training Results
============================================================
Initial loss: 45.678901
Final loss:   0.123456
Loss reduction: 99.73%

✓ Loss decreased successfully!
```

<!-- TODO: And Remove below -->

## Key Verification Points

✓ **Loss decreases** - This confirms gradients are flowing correctly
✓ **Parameters update** - Weight and bias values change during training
✓ **Predictions improve** - Trained model gives reasonable outputs
✓ **Gradients computed** - All parameters have non-zero gradients after backward()

## Troubleshooting

**ModuleNotFoundError: No module named 'synap'**
- Make sure you've compiled the C++ extension
- Add the build directory to PYTHONPATH

**Loss doesn't decrease**
- Check that backward() is computing gradients correctly
- Verify grad_values property returns non-zero values
- Ensure parameters are being updated (print before/after values)

**Segmentation fault**
- Check shared_ptr usage in backward pass
- Verify tensor lifecycle management
- Make sure grad tensors are allocated when needed

## What This Tests About Your Implementation

1. **Forward pass** - All operations (matmul, add, relu, mse) work correctly
2. **Backward pass** - Gradients propagate through the computation graph
3. **Topological sort** - Operations are executed in the right order during backprop
4. **Gradient accumulation** - Multiple backward passes accumulate gradients
5. **Parameter updates** - Gradients can be used to update weights and biases
6. **View operation** - Reshaping tensors maintains gradient flow
7. **Concat operation** - Merging outputs maintains gradient flow
