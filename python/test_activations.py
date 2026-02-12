import synap
import math

def assert_close(actual, expected, tol=1e-6):
    assert len(actual) == len(expected), f"Length mismatch: {actual} vs {expected}"
    for a, e in zip(actual, expected):
        assert abs(a - e) < tol, f"Value mismatch: {actual} vs {expected}"


def activation_tests():
    # Input tensor
    x = synap.Tensor([2, 3], requires_grad=True)
    x.set_values([-1.0, 0.0, 1.0, -2.0, 2.0, 3.0])

    # === ReLU ===
    y = synap.Tensor.relu(x)
    expected_forward = [0.0, 0.0, 1.0, 0.0, 2.0, 3.0]
    assert_close(synap.tensor_data(y), expected_forward)

    # Backward
    y.backward()
    expected_grad = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]  # d/dx ReLU
    assert_close(synap.tensor_data(x.grad), expected_grad)
    x.zero_grad()

    # === Sigmoid ===
    y = synap.Tensor.sigmoid(x)
    expected_forward = [1/(1+math.exp(1)),
                        0.5,
                        1/(1+math.exp(-1)),
                        1/(1+math.exp(2)),
                        1/(1+math.exp(-2)),
                        1/(1+math.exp(-3))]
    assert_close(synap.tensor_data(y), expected_forward)

    # Backward
    y.backward()
    expected_grad = [f*(1-f) for f in expected_forward]  # dy/dx = sigmoid*(1-sigmoid)
    assert_close(synap.tensor_data(x.grad), expected_grad)
    x.zero_grad()

    # === Tanh ===
    y = synap.Tensor.tanh(x)
    expected_forward = [math.tanh(v) for v in [-1, 0, 1, -2, 2, 3]]
    assert_close(synap.tensor_data(y), expected_forward)

    # Backward
    y.backward()
    expected_grad = [1 - f**2 for f in expected_forward]  # dy/dx = 1 - tanh^2(x)
    assert_close(synap.tensor_data(x.grad), expected_grad)
    x.zero_grad()


if __name__ == "__main__":
    activation_tests()
    print("All activation tests passed.")
