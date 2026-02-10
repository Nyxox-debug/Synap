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

    t3.backward()

    print("t2 grad values:", t2.grad_values)

    # Inspect tensor values if needed
    print("t3 values:", synap.tensor_data(t3))

    # Scalar tensor test
    scalar_a = synap.Tensor([1], requires_grad=True)
    scalar_b = synap.Tensor([1], requires_grad=True)
    scalar_c = synap.Tensor.add(scalar_a, scalar_b)

    scalar_c.backward()
    print("scalar_a.grad shape:", scalar_a.grad.shape())
    print("scalar_b.grad shape:", scalar_b.grad.shape())

def test_scalar_sink():
    t = synap.Tensor([2, 3], requires_grad=True)
    t.set_values([1, 2, 3, 4, 5, 6])

    s = synap.Tensor.sum(t)
    print("Sum value:", synap.tensor_data(s))  # should print [21.0]

    s.backward()
    print("Gradient of t after sum backward:", t.grad_values)  # all ones

if __name__ == "__main__":
    test_tensor_add_and_backward()
    test_scalar_sink()
