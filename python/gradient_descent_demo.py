"""
Minimal Gradient Descent Example

This demonstrates what your gradient descent test does, showing:
1. How parameters are updated based on gradients
2. How loss decreases over iterations
3. The expected behavior of your autodiff system

This is a conceptual demonstration - the actual test_gradient_descent.py 
will use your real Synap tensors and autodiff.
"""

import random
import math

class MockTensor:
    """Simplified tensor for demonstration purposes"""
    def __init__(self, values, requires_grad=False):
        self.values = values if isinstance(values, list) else [values]
        self.requires_grad = requires_grad
        self.grad = [0.0] * len(self.values) if requires_grad else None
    
    def zero_grad(self):
        if self.grad:
            self.grad = [0.0] * len(self.values)

def simple_linear_regression_demo():
    """
    Demonstrate gradient descent on y = w*x + b
    Target: y = 3*x + 2
    """
    print("=" * 60)
    print("Simple Linear Regression with Gradient Descent")
    print("=" * 60)
    print("Target function: y = 3*x + 2")
    print("We'll learn the parameters w and b\n")
    
    # Generate training data
    random.seed(42)
    data = []
    for _ in range(20):
        x = random.uniform(-5, 5)
        y = 3*x + 2 + random.uniform(-0.5, 0.5)  # True function with noise
        data.append((x, y))
    
    # Initialize parameters randomly
    w = random.uniform(-1, 1)  # weight
    b = random.uniform(-1, 1)  # bias
    
    print(f"Initial parameters: w = {w:.4f}, b = {b:.4f}\n")
    
    # Training parameters
    learning_rate = 0.01
    epochs = 100
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        grad_w_sum = 0.0
        grad_b_sum = 0.0
        
        # Process each data point
        for x, y_true in data:
            # Forward pass: y_pred = w*x + b
            y_pred = w * x + b
            
            # Compute loss: MSE = (y_pred - y_true)^2
            loss = (y_pred - y_true) ** 2
            epoch_loss += loss
            
            # Backward pass: compute gradients
            # d(loss)/d(y_pred) = 2 * (y_pred - y_true)
            d_loss_d_pred = 2 * (y_pred - y_true)
            
            # d(y_pred)/d(w) = x
            grad_w = d_loss_d_pred * x
            
            # d(y_pred)/d(b) = 1
            grad_b = d_loss_d_pred * 1
            
            # Accumulate gradients
            grad_w_sum += grad_w
            grad_b_sum += grad_b
        
        # Average the gradients
        grad_w_avg = grad_w_sum / len(data)
        grad_b_avg = grad_b_sum / len(data)
        
        # Update parameters: param = param - learning_rate * gradient
        w = w - learning_rate * grad_w_avg
        b = b - learning_rate * grad_b_avg
        
        # Calculate average loss
        avg_loss = epoch_loss / len(data)
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:8.4f}, w = {w:.4f}, b = {b:.4f}")
    
    print(f"\nFinal parameters: w = {w:.4f} (target: 3.0), b = {b:.4f} (target: 2.0)")
    print(f"Error in w: {abs(w - 3.0):.4f}")
    print(f"Error in b: {abs(b - 2.0):.4f}")
    
    # Test on new data
    print("\nTesting on new data points:")
    test_points = [(-2, 3*(-2)+2), (0, 2), (1, 5), (3, 11)]
    for x, y_expected in test_points:
        y_predicted = w * x + b
        error = abs(y_predicted - y_expected)
        print(f"  x={x:4.1f}: predicted={y_predicted:6.2f}, expected={y_expected:6.2f}, error={error:.2f}")


def neural_network_demo():
    """
    Demonstrate gradient descent on a simple 1-layer neural network
    Input: 2D, Hidden: 3 neurons, Output: 1D
    """
    print("\n" + "=" * 60)
    print("Neural Network Training with Gradient Descent")
    print("=" * 60)
    print("Architecture: 2 inputs -> 3 hidden (ReLU) -> 1 output")
    print("Task: Learn f(x1, x2) = x1 + 2*x2\n")
    
    # Generate training data
    random.seed(123)
    data = []
    for _ in range(30):
        x1 = random.uniform(-3, 3)
        x2 = random.uniform(-3, 3)
        y = x1 + 2*x2 + random.uniform(-0.3, 0.3)
        data.append(([x1, x2], y))
    
    # Initialize network parameters
    # Layer 1: 2 inputs -> 3 hidden neurons
    W1 = [[random.uniform(-0.5, 0.5) for _ in range(2)] for _ in range(3)]
    b1 = [random.uniform(-0.1, 0.1) for _ in range(3)]
    
    # Layer 2: 3 hidden -> 1 output
    W2 = [[random.uniform(-0.5, 0.5) for _ in range(3)]]
    b2 = [random.uniform(-0.1, 0.1)]
    
    learning_rate = 0.01
    epochs = 150
    
    print("Training...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Gradients (accumulated over batch)
        dW1 = [[0.0 for _ in range(2)] for _ in range(3)]
        db1 = [0.0 for _ in range(3)]
        dW2 = [[0.0 for _ in range(3)]]
        db2 = [0.0]
        
        for x_input, y_true in data:
            # Forward pass
            # Layer 1: h = ReLU(W1 @ x + b1)
            h_pre = [sum(W1[i][j] * x_input[j] for j in range(2)) + b1[i] for i in range(3)]
            h = [max(0, val) for val in h_pre]  # ReLU
            
            # Layer 2: y = W2 @ h + b2
            y_pred = sum(W2[0][i] * h[i] for i in range(3)) + b2[0]
            
            # Loss: MSE
            loss = (y_pred - y_true) ** 2
            epoch_loss += loss
            
            # Backward pass
            # Gradient of loss w.r.t. output
            d_loss = 2 * (y_pred - y_true)
            
            # Layer 2 gradients
            for i in range(3):
                dW2[0][i] += d_loss * h[i]
            db2[0] += d_loss
            
            # Backprop through layer 2
            d_h = [d_loss * W2[0][i] for i in range(3)]
            
            # Backprop through ReLU
            d_h_pre = [d_h[i] if h_pre[i] > 0 else 0 for i in range(3)]
            
            # Layer 1 gradients
            for i in range(3):
                for j in range(2):
                    dW1[i][j] += d_h_pre[i] * x_input[j]
                db1[i] += d_h_pre[i]
        
        # Average gradients
        n = len(data)
        for i in range(3):
            for j in range(2):
                dW1[i][j] /= n
            db1[i] /= n
        for i in range(3):
            dW2[0][i] /= n
        db2[0] /= n
        
        # Update parameters
        for i in range(3):
            for j in range(2):
                W1[i][j] -= learning_rate * dW1[i][j]
            b1[i] -= learning_rate * db1[i]
        for i in range(3):
            W2[0][i] -= learning_rate * dW2[0][i]
        b2[0] -= learning_rate * db2[0]
        
        # Print progress
        avg_loss = epoch_loss / len(data)
        if epoch % 30 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}")
    
    # Test the network
    print("\nTesting:")
    test_cases = [
        ([1, 1], 1 + 2*1),      # Expected: 3
        ([2, 0], 2 + 2*0),      # Expected: 2
        ([0, 3], 0 + 2*3),      # Expected: 6
        ([1, -1], 1 + 2*(-1)),  # Expected: -1
    ]
    
    for x_input, y_expected in test_cases:
        # Forward pass
        h_pre = [sum(W1[i][j] * x_input[j] for j in range(2)) + b1[i] for i in range(3)]
        h = [max(0, val) for val in h_pre]
        y_pred = sum(W2[0][i] * h[i] for i in range(3)) + b2[0]
        
        error = abs(y_pred - y_expected)
        print(f"  Input: {x_input} -> Predicted: {y_pred:6.2f}, Expected: {y_expected:6.2f}, Error: {error:.2f}")


def gradient_descent_concept():
    """
    Explain the concept of gradient descent
    """
    print("\n" + "=" * 60)
    print("How Gradient Descent Works")
    print("=" * 60)
    
    print("""
The gradient descent algorithm:

1. FORWARD PASS
   - Compute predictions using current parameters
   - Calculate loss (how wrong the predictions are)

2. BACKWARD PASS (Backpropagation)
   - Compute gradients: ∂Loss/∂param for each parameter
   - Use chain rule to propagate gradients backward through the network

3. PARAMETER UPDATE
   - Update each parameter: param_new = param_old - learning_rate × gradient
   - The gradient tells us which direction makes loss worse
   - We move in the opposite direction (hence the minus sign)

4. REPEAT
   - Do this for multiple epochs until loss converges

Key insight: The gradient ∂Loss/∂param tells us how much the loss changes
when we change that parameter. By moving parameters in the direction that
reduces loss, we gradually improve our model.

Your Synap framework automates step 2 (computing gradients) using autodiff!
The test_gradient_descent.py file verifies this works correctly by:
- Running the full training loop
- Checking that loss decreases (proving gradients are correct)
- Verifying the learned parameters are close to the true values
""")


if __name__ == "__main__":
    gradient_descent_concept()
    simple_linear_regression_demo()
    neural_network_demo()
    
    print("\n" + "=" * 60)
    print("This demonstrates what test_gradient_descent.py does")
    print("but using your actual Synap tensor autodiff system!")
    print("=" * 60)
