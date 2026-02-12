import synap

# --- Define network parameters ---
# Input: 3 features, hidden: 2 neurons, output: 2 classes
w1 = synap.Tensor([3, 2], requires_grad=True)
b1 = synap.Tensor([2], requires_grad=True)
w2 = synap.Tensor([2, 2], requires_grad=True)
b2 = synap.Tensor([2], requires_grad=True)
b2 = synap.Tensor([2], requires_grad=True)

# Random init for testing
w1.set_values([0.1, -0.2, 0.3, 0.4, -0.5, 0.2])
b1.set_values([0.0, 0.1])
w2.set_values([0.2, -0.1, 0.3, 0.1])
b2.set_values([0.0, -0.1])

# --- Helper functions ---
def linear(x, w, b):
    return synap.Tensor.add(synap.Tensor.matmul(x, w), b)

def relu(x):
    return synap.Tensor.relu(x)

# --- Dummy input data ---
x_test = synap.Tensor([2, 3], requires_grad=False)
x_test.set_values([
    1.0, 0.0, 1.0,
    0.0, 1.0, 1.0
])

# --- Forward pass ---
hidden = relu(linear(x_test, w1, b1))
logits = linear(hidden, w2, b2)

# Convert logits to predictions using argmax
preds = []
logits_data = synap.tensor_data(logits)
num_classes = 2
for i in range(0, len(logits_data), num_classes):
    row = logits_data[i:i+num_classes]
    pred_class = row.index(max(row))
    preds.append(pred_class)

print("Predictions:", preds)
