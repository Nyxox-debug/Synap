# MNIST Digit Recognition Test

Train a neural network to recognize handwritten digits using your Synap autodiff framework!

## Files

- **test_mnist_simple.py** - Recommended! Works with or without MNIST data
- **test_mnist.py** - Full version with batch training
- **nn.py** - Neural network module (Module, Neuron, Layer, MLP)

## Quick Start

### Option 1: Run with Synthetic Data (No Setup Required)

```bash
python test_mnist_simple.py
```

This will automatically create synthetic digit-like data and train on it. Perfect for testing your autodiff implementation!

### Option 2: Run with Real MNIST Data

1. **Download MNIST dataset** from http://yann.lecun.com/exdb/mnist/

   You need these 4 files:
   - `train-images-idx3-ubyte` (9.9 MB)
   - `train-labels-idx1-ubyte` (28.9 KB)
   - `t10k-images-idx3-ubyte` (1.6 MB)
   - `t10k-labels-idx1-ubyte` (4.5 KB)

   Extract them (if gzipped) and place in the same directory as the test script.

2. **Run the test:**
   ```bash
   python test_mnist_simple.py  # Uses real data if available
   # or
   python test_mnist.py         # Full version with mini-batch training
   ```

## What It Does

### Architecture
```
Input Layer:    784 neurons (28×28 pixel image flattened)
Hidden Layer:   64 neurons (or 128 in full version)
Output Layer:   10 neurons (one per digit 0-9)
```

### Training Process

1. **Load Data** - MNIST images or synthetic patterns
2. **Forward Pass** - Compute predictions for each image
3. **Loss Calculation** - Softmax cross-entropy loss
4. **Backward Pass** - Compute gradients via autodiff
5. **Update Weights** - Gradient descent: `w = w - lr * grad`
6. **Repeat** - Train for multiple epochs

### Expected Output

```
============================================================
Digit Recognition with Gradient Descent
============================================================

✓ Loaded real MNIST data
  Training: 500 images
  Testing: 100 images

Model architecture: 784 -> 64 -> 10

Learning rate: 0.05
Epochs: 10

============================================================
Training
============================================================
Epoch  1/10: Loss = 2.3456, Accuracy = 15.2%
Epoch  2/10: Loss = 1.8234, Accuracy = 35.6%
Epoch  3/10: Loss = 1.4567, Accuracy = 52.4%
...
Epoch 10/10: Loss = 0.6543, Accuracy = 82.8%

============================================================
Testing
============================================================
Sample predictions:
  ✓ Predicted: 7, Actual: 7
  ✓ Predicted: 2, Actual: 2
  ✗ Predicted: 9, Actual: 4
  ✓ Predicted: 1, Actual: 1
  ...

Test Accuracy: 78/100 = 78.0%

============================================================
```

## What This Tests

✅ **Multi-class classification** - 10 output classes (digits 0-9)  
✅ **Softmax cross-entropy loss** - Proper loss for classification  
✅ **Larger networks** - 784 → 64 → 10 architecture  
✅ **Real-world problem** - Handwriting recognition!  
✅ **Accuracy metrics** - Not just loss, but actual prediction accuracy  
✅ **Gradient descent convergence** - Loss decreases, accuracy increases  

## Customization

You can adjust these parameters in the code:

```python
learning_rate = 0.05    # How fast the model learns
epochs = 10             # Number of training passes
hidden_size = 64        # Number of neurons in hidden layer
```

For `test_mnist.py`:
```python
batch_size = 10         # Mini-batch size for training
```

## Troubleshooting

**Low accuracy with synthetic data**  
- Normal! Synthetic data is much simpler than real handwriting
- You should still see loss decreasing and accuracy improving
- With real MNIST, expect 70-85% accuracy after 10 epochs

**Out of memory**  
- Reduce the number of training samples
- Reduce `hidden_size` 
- Use smaller batches in `test_mnist.py`

**Training is slow**  
- Normal for 784-dimensional inputs!
- Each epoch processes 500-1000 images
- Consider reducing training samples for faster testing

**Accuracy not improving**  
- Try different learning rates (0.01, 0.05, 0.1)
- Train for more epochs
- Check that gradients are being computed (should see loss decreasing)

## Comparison to Your Simple Test

Your original `test_simple_gd.py`:
- 4 input dimensions
- 1 training example
- Simple MSE loss

This MNIST test:
- 784 input dimensions (28×28 images)
- 500-1000 training examples
- Softmax cross-entropy loss
- Real classification problem!

Both use the same gradient descent principle:
```python
for param in model.parameters():
    param -= learning_rate * gradient
```

But MNIST demonstrates your framework working on a real machine learning problem!

## Next Steps

Once this works, you have a fully functional deep learning framework! 🎉

You can try:
- Different architectures (more layers, more neurons)
- Different datasets (CIFAR-10, Fashion-MNIST)
- Adding momentum, Adam optimizer
- Convolutional layers for better accuracy
- Regularization techniques (dropout, L2)
