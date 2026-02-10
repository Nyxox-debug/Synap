import synap

def create_tensor(shape, values=None, requires_grad=True):
    t = synap.Tensor(shape, requires_grad=requires_grad)
    if values is not None:
        t.set_values(values)
    return t

def print_tensor_info(name, t):
    print(f"{name} shape: {t.shape()}, values: {synap.tensor_data(t)}, grad: {t.grad_values}")

# Binary Operation + Backward Test
def test_binary_ops():
    print("=== Binary Operations + Backward Test ===")
    
    a_vals = [1, 2, 3, 4]
    b_vals = [2, 3, 4, 5]
    
    t1 = create_tensor([2, 2], a_vals)
    t2 = create_tensor([2, 2], b_vals)
    
    ops = {
        "add": synap.Tensor.add,
        "sub": synap.Tensor.sub,
        "mul": synap.Tensor.mul,
        "div": synap.Tensor.div
    }

    for name, op in ops.items():
        print(f"\n--- Testing {name} ---")
        t_out = op(t1, t2)
        t_out.backward()
        print_tensor_info("t1", t1)
        print_tensor_info("t2", t2)
        print_tensor_info(f"t_out ({name})", t_out)
        t1.zero_grad()
        t2.zero_grad()

# Scalar Sink (sum)
def test_scalar_sink_sum():
    print("\n=== Scalar Sink Sum Test ===")
    
    t = create_tensor([2, 3], [1, 2, 3, 4, 5, 6])
    s = synap.Tensor.sum(t)
    
    print("Sum value:", synap.tensor_data(s))  # [21.0]
    
    s.backward()
    print("Gradient of t after sum backward:", t.grad_values)  # all ones

# Mean Test
def test_mean_operation():
    print("\n=== Mean Operation Test ===")
    
    t = create_tensor([2, 2], [1, 2, 3, 4])
    m = synap.Tensor.mean(t)
    
    print("Mean value:", synap.tensor_data(m))  # [2.5]
    
    m.backward()
    print("Gradient of t after mean backward:", t.grad_values)  
    # should be [0.25, 0.25, 0.25, 0.25]

# Linear Operations Test
def lineara_ops():
    print("\n=== Linear Operations Test ===")
    t = synap.Tensor([2, 3])
    t.set_values([1, 2, 3, 4, 5, 6])
    t_t = synap.Tensor.transpose(t)
    print("Original:", synap.tensor_data(t))   # [1,2,3,4,5,6]
    print("Transposed:", synap.tensor_data(t_t))  # should be [1,4,2,5,3,6] if row-major

    a = synap.Tensor([2,3], requires_grad=True)
    b = synap.Tensor([2,4], requires_grad=True)
    a.set_values([1,2,3,4,5,6])
    b.set_values([1,2,3,4,5,6,7,8])

    c = synap.Tensor.matmul(synap.Tensor.transpose(a), b)
    c.backward()
    print("a grad:", synap.tensor_data(a.grad))
    print("b grad:", synap.tensor_data(b.grad))

# Main
if __name__ == "__main__":
    test_binary_ops()
    test_scalar_sink_sum()
    test_mean_operation()
    lineara_ops()
