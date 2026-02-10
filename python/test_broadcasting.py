import synap

def scalar_broadcasting_tests():
    # Setup tensor and scalar
    t = synap.Tensor([2, 2], requires_grad=True)
    t.set_values([1, 2, 3, 4])

    s = synap.Tensor([1], requires_grad=True)
    s.set_values([10])

    print("=== ADD ===")
    o = synap.Tensor.add(t, s)
    print("Output:", synap.tensor_data(o))  # [11,12,13,14]
    o.backward()
    print("t grad:", synap.tensor_data(t.grad))  # [1,1,1,1]
    print("s grad:", synap.tensor_data(s.grad))  # [4]
    t.zero_grad()
    s.zero_grad()

    print("\n=== SUB ===")
    o = synap.Tensor.sub(t, s)
    print("Output:", synap.tensor_data(o))  # [-9,-8,-7,-6]
    o.backward()
    print("t grad:", synap.tensor_data(t.grad))  # [1,1,1,1]
    print("s grad:", synap.tensor_data(s.grad))  # [-4]
    t.zero_grad()
    s.zero_grad()

    print("\n=== MUL ===")
    o = synap.Tensor.mul(t, s)
    print("Output:", synap.tensor_data(o))  # [10,20,30,40]
    o.backward()
    print("t grad:", synap.tensor_data(t.grad))  # [10,10,10,10]
    print("s grad:", synap.tensor_data(s.grad))  # [10+20+30+40 = 100]
    t.zero_grad()
    s.zero_grad()

    print("\n=== DIV ===")
    o = synap.Tensor.div(t, s)
    print("Output:", synap.tensor_data(o))  # [0.1,0.2,0.3,0.4]
    o.backward()
    print("t grad:", synap.tensor_data(t.grad))  # [0.1,0.1,0.1,0.1]
    print("s grad:", synap.tensor_data(s.grad))  # [-sum(t_i / s^2) = -0.1-0.2-0.3-0.4 = -1.0]

if __name__ == "__main__":
    scalar_broadcasting_tests()
