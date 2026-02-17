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
    x = synap.Tensor([1, 2], requires_grad=True)
    x.set_values([1, 2])
    lyr = nn.Layer(nin=2, nout=4)
    out = lyr(x)

    assert isinstance(out, synap.Tensor), "Layer output should be a Tensor"
    assert out.shape() == [4], f"Unexpected output shape: {out.shape()}"
    print(f"test_layer {synap.tensor_data(out)}")

def test_MLP():
    x = synap.Tensor([2, 2], requires_grad=True)
    x.set_values([1, 2, 3, 4])
    # NOTE: Demonstrating View method
    x = x.view([1, 4])
    mlp = nn.MLP(4, [8, 8, 4, 1])
    out = mlp(x)

    print("len of MLP params: ", len(mlp.parameters()))
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

def test_parameters():
    x = synap.Tensor([1, 4], requires_grad=True)
    x.set_values([1, 2, 3, 4])
    m = nn.MLP(4, [4, 4, 4, 4, 1])

    # print(m.parameters())
    # print("test_parameters()")
    # for li, layer in enumerate(m.layers):
    #     print(f"Layer {li}:")
    #     for ni, neuron in enumerate(layer.neurons):
    #         print(f"  Neuron {ni} weights: {synap.tensor_data(neuron.w)}")
    #         print(f"  Neuron {ni} bias: {synap.tensor_data(neuron.b)}")

    assert len(m.parameters()) == 34, f"Unexpected paramas length: {len(m.parameters())}"
    print(f"test_parameters {len(m.parameters())}")

if __name__ == "__main__":
    test_neuron()
    test_layer()
    test_MLP()
    test_MLP_5x5()
    test_parameters()
    print("All tests passed!")
