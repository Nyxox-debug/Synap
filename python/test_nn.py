import random
import synap
from nn import MLP

# Seed for reproducibility
random.seed(42)

# Create a small MLP: 3 input features → 4 hidden neurons → 2 output neurons
model = MLP(3, [4, 2])
print(model)

# Create a dummy input tensor (shape [3])
x = synap.Tensor([3], requires_grad=False)
x.set_values([0.5, -1.2, 3.3])

# Forward pass
output = model(x)
print("Output tensor data:", synap.tensor_data(output))

# Dummy target for MSE loss
target = synap.Tensor([2], requires_grad=False)
target.set_values([1.0, 0.0])  # example 2-class target

# Compute loss using your framework's MSE
loss = synap.Tensor.mse(output, target)
print("Loss:", synap.tensor_data(loss))

# Backward pass
loss.backward()
print("Gradients for first layer weights:")
for i, neuron in enumerate(model.layers[0].neurons):
    print(f"Neuron {i} grad w:", synap.tensor_data(neuron.w))
    print(f"Neuron {i} grad b:", synap.tensor_data(neuron.b))
