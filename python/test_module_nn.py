import nn
import synap

def test_neuron():
    x = synap.Tensor([1, 4], requires_grad=True)
    x.set_values([1, 2, 3, 4])
    n = nn.Neuron(4)
    out = n(x)

    assert isinstance(out, synap.Tensor), "Neuron output should be a Tensor"
    assert out.shape() == [1], f"Unexpected output shape: {out.shape()}"
    print(f"test_neuron {synap.tensor_data(out)}")

def test_layer():
    x = synap.Tensor([2, 2], requires_grad=True)
    x.set_values([1, 2, 3, 4])
    lyr = nn.Layer(nin=2, nout=4)
    out = lyr(x)

    assert isinstance(out, synap.Tensor), "Layer output should be a Tensor"
    assert out.shape() == [8], f"Unexpected output shape: {out.shape()}"
    print(f"test_layer {synap.tensor_data(out)}")

def test_MLP():
    x = synap.Tensor([2, 2], requires_grad=True)
    x.set_values([1, 2, 3, 4])
    x = x.view([1, 4])
    mlp = nn.MLP(4, [8, 8, 4, 1])
    out = mlp(x)

    assert isinstance(out, synap.Tensor), "MLP output should be a Tensor"
    assert out.shape() == [1], f"Unexpected output shape: {out.shape()}"
    print(f"test_MLP {synap.tensor_data(out)}")

def test_MLP_5x5():
    x = synap.Tensor([5, 5], requires_grad=True)
    x.set_values([
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.5, 0.4, 0.3, 0.2, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.9, 0.8, 0.7, 0.6, 0.5,
        0.0, 0.2, 0.4, 0.6, 0.8
    ])
    x = x.view([1, 25])
    mlp = nn.MLP(25, [16, 8, 1])
    out = mlp(x)

    assert isinstance(out, synap.Tensor), "MLP output should be a Tensor"
    assert out.shape() == [1], f"Unexpected output shape: {out.shape()}"
    print(f"test_MLP_5x5 {synap.tensor_data(out)}")

if __name__ == "__main__":
    test_neuron()
    test_layer()
    test_MLP()
    test_MLP_5x5()
    print("All tests passed!")
