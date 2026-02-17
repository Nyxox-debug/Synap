import synap
import math
import numpy as np

def assert_close(actual, expected, tol=1e-6):
    assert len(actual) == len(expected), f"Length mismatch: {actual} vs {expected}"
    for a, e in zip(actual, expected):
        assert abs(a - e) < tol, f"Value mismatch: {actual} vs {expected}"

def mse_loss_test():
    # Forward
    pred = synap.Tensor([2, 2], requires_grad=True)
    pred.set_values([1.0, 2.0, 3.0, 4.0])
    target = synap.Tensor([2, 2])
    target.set_values([0.0, 2.0, 1.0, 4.0])

    loss = synap.Tensor.mse(pred, target)
    expected_loss = ((1-0)**2 + (2-2)**2 + (3-1)**2 + (4-4)**2) / 4  # 1.25
    assert math.isclose(synap.tensor_data(loss)[0], expected_loss), \
        f"MSE forward mismatch: {synap.tensor_data(loss)} vs {expected_loss}"

    # Backward
    loss.backward()
    # Grad: 2*(pred-target)/n
    expected_grad = [0.5, 0.0, 1.0, 0.0]
    assert_close(synap.tensor_data(pred.grad), expected_grad)
    pred.zero_grad()

if __name__ == "__main__":
    mse_loss_test()
    print("All Phase 6 tests passed.")
