# test_synap_tensor.py
import synap

def test_tensor_add_and_backward():
    print("=== Tensor Add + Backward Test ===")

    # Create two tensors
    t1 = synap.Tensor([2, 2], requires_grad=True)
    t2 = synap.Tensor([2, 2], requires_grad=True)

    # Add them using the static method
    t3 = synap.Tensor.add(t1, t2)

    print("t1 shape:", t1.shape())
    print("t2 shape:", t2.shape())
    print("t3 shape:", t3.shape())

    # Run backward on t3 (will raise error for non-scalar)
    try:
        t3.backward()
    except RuntimeError as e:
        print("Expected error for non-scalar backward:", e)

    # Scalar tensor test
    scalar_a = synap.Tensor([1], requires_grad=True)
    scalar_b = synap.Tensor([1], requires_grad=True)
    scalar_c = synap.Tensor.add(scalar_a, scalar_b)

    scalar_c.backward()
    print("scalar_a.grad shape:", scalar_a.grad.shape())
    print("scalar_b.grad shape:", scalar_b.grad.shape())

if __name__ == "__main__":
    test_tensor_add_and_backward()
