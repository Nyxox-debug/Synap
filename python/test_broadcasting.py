import synap


def assert_close(actual, expected, tol=1e-6):
    assert len(actual) == len(expected), f"Length mismatch: {actual} vs {expected}"
    for a, e in zip(actual, expected):
        assert abs(a - e) < tol, f"Value mismatch: {actual} vs {expected}"


def scalar_broadcasting_tests():
    t = synap.Tensor([2, 2], requires_grad=True)
    t.set_values([1, 2, 3, 4])

    s = synap.Tensor([1], requires_grad=True)
    s.set_values([10])

    # === ADD ===
    o = synap.Tensor.add(t, s)
    assert_close(synap.tensor_data(o), [11, 12, 13, 14])
    o.backward()
    assert_close(synap.tensor_data(t.grad), [1, 1, 1, 1])
    assert_close(synap.tensor_data(s.grad), [4])
    t.zero_grad()
    s.zero_grad()

    # === SUB ===
    o = synap.Tensor.sub(t, s)
    assert_close(synap.tensor_data(o), [-9, -8, -7, -6])
    o.backward()
    assert_close(synap.tensor_data(t.grad), [1, 1, 1, 1])
    assert_close(synap.tensor_data(s.grad), [-4])
    t.zero_grad()
    s.zero_grad()

    # === MUL ===
    o = synap.Tensor.mul(t, s)
    assert_close(synap.tensor_data(o), [10, 20, 30, 40])
    o.backward()
    assert_close(synap.tensor_data(t.grad), [10, 10, 10, 10])
    assert_close(synap.tensor_data(s.grad), [1 + 2 + 3 + 4])  # 10
    t.zero_grad()
    s.zero_grad()

    # === DIV ===
    o = synap.Tensor.div(t, s)
    assert_close(synap.tensor_data(o), [0.1, 0.2, 0.3, 0.4])
    o.backward()
    assert_close(synap.tensor_data(t.grad), [0.1, 0.1, 0.1, 0.1])
    # d/ds sum(t/s) = -sum(t) / s^2 = -(1+2+3+4)/100 = -10/100 = -0.1
    assert_close(synap.tensor_data(s.grad), [-0.1])
    t.zero_grad()
    s.zero_grad()


def row_column_broadcasting_tests():
    A = synap.Tensor([2, 3], requires_grad=True)
    A.set_values([
        1, 2, 3,
        4, 5, 6
    ])

    row = synap.Tensor([3], requires_grad=True)
    row.set_values([10, 20, 30])

    col = synap.Tensor([2, 1], requires_grad=True)
    col.set_values([100, 200])

    # === ROW BROADCAST ADD ===
    o = synap.Tensor.add(A, row)
    assert_close(
        synap.tensor_data(o),
        [
            11, 22, 33,
            14, 25, 36
        ]
    )
    o.backward()
    assert_close(synap.tensor_data(A.grad), [1, 1, 1, 1, 1, 1])
    assert_close(synap.tensor_data(row.grad), [2, 2, 2])
    A.zero_grad()
    row.zero_grad()

    # === COLUMN BROADCAST ADD ===
    o = synap.Tensor.add(A, col)
    assert_close(
        synap.tensor_data(o),
        [
            101, 102, 103,
            204, 205, 206
        ]
    )
    o.backward()
    assert_close(synap.tensor_data(A.grad), [1, 1, 1, 1, 1, 1])
    assert_close(synap.tensor_data(col.grad), [3, 3])
    A.zero_grad()
    col.zero_grad()


if __name__ == "__main__":
    scalar_broadcasting_tests()
    row_column_broadcasting_tests()
    print("All broadcasting tests passed.")
