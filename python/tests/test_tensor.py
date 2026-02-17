import synap


def assert_close(actual, expected, tol=1e-6):
    assert len(actual) == len(expected), f"Length mismatch: {actual} vs {expected}"
    for a, e in zip(actual, expected):
        assert abs(a - e) < tol, f"Value mismatch: {actual} vs {expected}"


def create_tensor(shape, values=None, requires_grad=True):
    t = synap.Tensor(shape, requires_grad=requires_grad)
    if values is not None:
        t.set_values(values)
    return t


# =========================
# Binary Operation + Backward Test
# =========================
def test_binary_ops():
    a_vals = [1, 2, 3, 4]
    b_vals = [2, 3, 4, 5]

    t1 = create_tensor([2, 2], a_vals)
    t2 = create_tensor([2, 2], b_vals)

    # ADD
    out = synap.Tensor.add(t1, t2)
    assert_close(synap.tensor_data(out), [3, 5, 7, 9])
    out.backward()
    assert_close(synap.tensor_data(t1.grad), [1, 1, 1, 1])
    assert_close(synap.tensor_data(t2.grad), [1, 1, 1, 1])
    t1.zero_grad(); t2.zero_grad()

    # SUB
    out = synap.Tensor.sub(t1, t2)
    assert_close(synap.tensor_data(out), [-1, -1, -1, -1])
    out.backward()
    assert_close(synap.tensor_data(t1.grad), [1, 1, 1, 1])
    assert_close(synap.tensor_data(t2.grad), [-1, -1, -1, -1])
    t1.zero_grad(); t2.zero_grad()

    # MUL
    out = synap.Tensor.mul(t1, t2)
    assert_close(synap.tensor_data(out), [2, 6, 12, 20])
    out.backward()
    assert_close(synap.tensor_data(t1.grad), b_vals)
    assert_close(synap.tensor_data(t2.grad), a_vals)
    t1.zero_grad(); t2.zero_grad()

    # DIV
    out = synap.Tensor.div(t1, t2)
    expected = [1/2, 2/3, 3/4, 4/5]
    assert_close(synap.tensor_data(out), expected)
    out.backward()

    # d(a/b)/da = 1/b
    assert_close(synap.tensor_data(t1.grad), [1/2, 1/3, 1/4, 1/5])

    # d(a/b)/db = -a/b^2
    expected_b_grad = [
        -1/(2**2),
        -2/(3**2),
        -3/(4**2),
        -4/(5**2),
    ]
    assert_close(synap.tensor_data(t2.grad), expected_b_grad)

    t1.zero_grad(); t2.zero_grad()


# =========================
# Scalar Sink (Sum)
# =========================
def test_scalar_sink_sum():
    t = create_tensor([2, 3], [1, 2, 3, 4, 5, 6])
    s = synap.Tensor.sum(t)

    assert_close(synap.tensor_data(s), [21])

    s.backward()
    assert_close(synap.tensor_data(t.grad), [1, 1, 1, 1, 1, 1])


# =========================
# Mean Test
# =========================
def test_mean_operation():
    t = create_tensor([2, 2], [1, 2, 3, 4])
    m = synap.Tensor.mean(t)

    assert_close(synap.tensor_data(m), [2.5])

    m.backward()
    assert_close(synap.tensor_data(t.grad), [0.25, 0.25, 0.25, 0.25])


# =========================
# Linear Ops: Transpose + Matmul
# =========================
def test_linear_ops():
    # Transpose check
    t = synap.Tensor([2, 3])
    t.set_values([1, 2, 3, 4, 5, 6])
    t_t = synap.Tensor.transpose(t)

    # row-major transpose of:
    # [1 2 3
    #  4 5 6]
    # ->
    # [1 4
    #  2 5
    #  3 6]
    assert_close(synap.tensor_data(t_t), [1, 4, 2, 5, 3, 6])

    # Matmul backward sanity test
    a = synap.Tensor([2, 3], requires_grad=True)
    b = synap.Tensor([2, 4], requires_grad=True)

    a.set_values([1, 2, 3, 4, 5, 6])
    b.set_values([1, 2, 3, 4, 5, 6, 7, 8])

    # (a^T) shape: 3x2
    # b shape: 2x4
    # result: 3x4
    c = synap.Tensor.matmul(synap.Tensor.transpose(a), b)

    # Just check shape and gradient propagation sanity
    assert c.shape() == [3, 4]

    c.backward()

    # If backward ran, gradients must exist and match tensor sizes
    assert len(synap.tensor_data(a.grad)) == 6
    assert len(synap.tensor_data(b.grad)) == 8


if __name__ == "__main__":
    test_binary_ops()
    test_scalar_sink_sum()
    test_mean_operation()
    test_linear_ops()
    print("All Tensor tests passed.")
