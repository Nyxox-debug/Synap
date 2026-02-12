import synap
from nn import MLP, SGD

# Create model
model = MLP([3, 16, 2])

# Dummy input (batch_size=1, features=3)
x = synap.Tensor([1, 3], requires_grad=False)
x.set_values([0.5, -1.2, 3.3])

# Dummy target (classification: 2 classes, one-hot)
target = synap.Tensor([1, 2], requires_grad=False)
target.set_values([1.0, 0.0])

# Forward
logits = model(x)
loss = synap.Tensor.softmax_cross_entropy(logits, target)

# Backward
loss.backward()

# Update
opt = SGD(model.parameters(), lr=0.01)
opt.step()
opt.zero_grad()

print("Logits:", synap.tensor_data(logits))
print("Loss:", synap.tensor_data(loss))
