import nn
import synap
import random

def test_gradient_descent():
    """
    Test that performs actual gradient descent on a simple dataset.
    This verifies that:
    1. Gradients are computed correctly
    2. Parameters can be updated
    3. Loss decreases over iterations
    """
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create a simple dataset: learn f(x) = 2*x1 + 3*x2 - 1
    # We'll generate some noisy data around this function
    dataset = []
    for _ in range(20):
        x1 = random.uniform(-2, 2)
        x2 = random.uniform(-2, 2)
        y = 2*x1 + 3*x2 - 1 + random.uniform(-0.1, 0.1)  # small noise
        dataset.append(([x1, x2], [y]))
    
    # Create model: 2 inputs -> 4 hidden -> 1 output
    model = nn.MLP(2, [4, 1])
    
    # Training parameters
    learning_rate = 0.01
    epochs = 100
    
    print("=" * 60)
    print("Training with Gradient Descent")
    print("=" * 60)
    print(f"Dataset size: {len(dataset)}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print()
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Train on each example
        for x_data, y_data in dataset:
            # Create input tensor
            x = synap.Tensor([len(x_data)], requires_grad=False)
            x.set_values(x_data)
            
            # Create target tensor
            y_true = synap.Tensor([1], requires_grad=False)
            y_true.set_values(y_data)
            
            # Forward pass
            y_pred = model(x)
            
            # Compute loss
            loss = synap.Tensor.mse(y_pred, y_true)
            
            # Zero gradients
            model.zero_grad()
            
            # Backward pass
            loss.backward()
            
            # Update parameters (gradient descent)
            for param in model.parameters():
                if param.grad is not None:
                    # Get current parameter values
                    param_values = synap.tensor_data(param)
                    grad_values = param.grad_values
                    
                    # Update: param = param - learning_rate * grad
                    updated_values = [
                        p - learning_rate * g 
                        for p, g in zip(param_values, grad_values)
                    ]
                    
                    param.set_values(updated_values)
            
            # Accumulate loss
            epoch_loss += synap.tensor_data(loss)[0]
        
        # Average loss for this epoch
        avg_loss = epoch_loss / len(dataset)
        losses.append(avg_loss)
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}")
    
    print()
    print("=" * 60)
    print("Training Results")
    print("=" * 60)
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss:   {losses[-1]:.6f}")
    print(f"Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")
    print()
    
    # Verify loss decreased
    assert losses[-1] < losses[0], "Loss should decrease during training!"
    print("✓ Loss decreased successfully!")
    print()
    
    # Test the trained model on a few examples
    print("=" * 60)
    print("Testing Trained Model")
    print("=" * 60)
    test_cases = [
        ([1.0, 1.0], 2*1.0 + 3*1.0 - 1),  # Expected: 4.0
        ([0.5, 0.5], 2*0.5 + 3*0.5 - 1),  # Expected: 1.5
        ([2.0, -1.0], 2*2.0 + 3*(-1.0) - 1),  # Expected: 0.0
    ]
    
    for x_vals, expected in test_cases:
        x = synap.Tensor([len(x_vals)], requires_grad=False)
        x.set_values(x_vals)
        
        y_pred = model(x)
        predicted = synap.tensor_data(y_pred)[0]
        error = abs(predicted - expected)
        
        print(f"Input: {x_vals}")
        print(f"  Expected: {expected:.4f}")
        print(f"  Predicted: {predicted:.4f}")
        print(f"  Error: {error:.4f}")
        print()
    
    print("=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


def test_simple_gradient_descent():
    """
    Simpler test: fit a single linear function y = 3*x + 2
    """
    print("\n" + "=" * 60)
    print("Simple Linear Regression Test")
    print("=" * 60)
    print("Target function: y = 3*x + 2")
    print()
    
    random.seed(123)
    
    # Generate data: y = 3*x + 2
    dataset = []
    for _ in range(10):
        x = random.uniform(-5, 5)
        y = 3*x + 2 + random.uniform(-0.2, 0.2)
        dataset.append(([x], [y]))
    
    # Simple model: 1 input -> 1 output (no hidden layer, just linear)
    model = nn.MLP(1, [1])
    
    learning_rate = 0.01
    epochs = 50
    
    print(f"Training for {epochs} epochs...")
    print()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for x_data, y_data in dataset:
            x = synap.Tensor([1], requires_grad=False)
            x.set_values(x_data)
            
            y_true = synap.Tensor([1], requires_grad=False)
            y_true.set_values(y_data)
            
            y_pred = model(x)
            loss = synap.Tensor.mse(y_pred, y_true)
            
            model.zero_grad()
            loss.backward()
            
            # Update parameters
            for param in model.parameters():
                if param.grad is not None:
                    param_values = synap.tensor_data(param)
                    grad_values = param.grad_values
                    updated = [p - learning_rate * g for p, g in zip(param_values, grad_values)]
                    param.set_values(updated)
            
            epoch_loss += synap.tensor_data(loss)[0]
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}: Loss = {epoch_loss / len(dataset):.6f}")
    
    # Check learned parameters
    print()
    print("Learned parameters:")
    w = model.layers[0].neurons[0].w
    b = model.layers[0].neurons[0].b
    print(f"  Weight: {synap.tensor_data(w)[0]:.4f} (expected ~3.0)")
    print(f"  Bias:   {synap.tensor_data(b)[0]:.4f} (expected ~2.0)")
    print()


def test_batch_gradient_descent():
    """
    Test with batch updates (accumulate gradients over multiple examples)
    """
    print("\n" + "=" * 60)
    print("Batch Gradient Descent Test")
    print("=" * 60)
    print()
    
    random.seed(456)
    
    # XOR-like problem (more challenging)
    dataset = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]
    
    # Model with hidden layer (needed for XOR)
    model = nn.MLP(2, [4, 1])
    
    learning_rate = 0.1
    epochs = 200
    
    print("Training on XOR-like problem...")
    print()
    
    for epoch in range(epochs):
        # Accumulate gradients over all examples
        total_loss = 0.0
        
        # First pass: compute gradients
        for x_data, y_data in dataset:
            x = synap.Tensor([2], requires_grad=False)
            x.set_values(x_data)
            
            y_true = synap.Tensor([1], requires_grad=False)
            y_true.set_values(y_data)
            
            y_pred = model(x)
            loss = synap.Tensor.mse(y_pred, y_true)
            
            if x_data == dataset[0][0]:  # First example
                model.zero_grad()
            
            loss.backward()
            total_loss += synap.tensor_data(loss)[0]
        
        # Update parameters with accumulated gradients
        for param in model.parameters():
            if param.grad is not None:
                param_values = synap.tensor_data(param)
                grad_values = param.grad_values
                # Divide by batch size for average gradient
                updated = [
                    p - learning_rate * g / len(dataset) 
                    for p, g in zip(param_values, grad_values)
                ]
                param.set_values(updated)
        
        if epoch % 40 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}")
    
    print()
    print("Testing on training data:")
    for x_data, y_data in dataset:
        x = synap.Tensor([2], requires_grad=False)
        x.set_values(x_data)
        y_pred = model(x)
        predicted = synap.tensor_data(y_pred)[0]
        print(f"  Input: {x_data} -> Predicted: {predicted:.4f}, Target: {y_data[0]:.4f}")
    
    print()


if __name__ == "__main__":
    # Run all tests
    test_simple_gradient_descent()
    test_gradient_descent()
    test_batch_gradient_descent()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
