import synap
import math
import numpy as np

def assert_close(actual, expected, tol=1e-6):
    assert len(actual) == len(expected), f"Length mismatch: {actual} vs {expected}"
    for a, e in zip(actual, expected):
        assert abs(a - e) < tol, f"Value mismatch: {actual} vs {expected}"

def softmax_cross_entropy_test():
    logits = synap.Tensor([2, 3], requires_grad=True)
    logits.set_values([1.0, 2.0, 3.0,
                       1.0, 2.0, 3.0])
    targets = synap.Tensor([2, 3])
    targets.set_values([0.0, 0.0, 1.0,
                        0.0, 1.0, 0.0])

    loss = synap.Tensor.softmax_cross_entropy(logits, targets)

    # Match C++ computation exactly
    l = np.array([[1,2,3],[1,2,3]], dtype=np.float32)
    t = np.array([[0,0,1],[0,1,0]], dtype=np.float32)

    # Shift logits
    l_shifted = l - np.max(l, axis=1, keepdims=True)
    exp_logits = np.exp(l_shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Compute cross-entropy row-wise, then mean over batch
    row_loss = -np.sum(t * np.log(probs), axis=1)
    expected_loss = np.mean(row_loss)  # This should now match C++ output

    # Forward assertion
    assert math.isclose(synap.tensor_data(loss)[0], expected_loss, rel_tol=1e-6), \
        f"SCE forward mismatch: {synap.tensor_data(loss)} vs {expected_loss}"

    # Backward: gradient = (probs - targets)/batch
    expected_grad = ((probs - t) / l.shape[0]).flatten().tolist()
    loss.backward()
    assert_close(synap.tensor_data(logits.grad), expected_grad)
    logits.zero_grad()

if __name__ == "__main__":
    softmax_cross_entropy_test()
    print("Softmax Cross-Entropy test passed.")
