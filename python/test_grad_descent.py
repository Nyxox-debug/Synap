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
        
        # Update parameters (gradient descent step)
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
