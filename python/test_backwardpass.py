import nn
import synap


def test_backwardpass():
    x = synap.Tensor([1, 4], requires_grad=True)
    x.set_values([1, 2, 3, 4])
    m = nn.MLP(4, [4, 4, 4, 4, 1])
    out = m(x)

    # print(m.parameters())
    for li, layer in enumerate(m.layers):
        print(f"Layer {li}:")
        for ni, neuron in enumerate(layer.neurons):
            print(f"  Neuron {ni} weights: {synap.tensor_data(neuron.w)}")
            print(f"  Neuron {ni} bias: {synap.tensor_data(neuron.b)}")

    print(len(m.parameters()))
    assert isinstance(out, synap.Tensor), "Neuron output should be a Tensor"
    assert out.shape() == [1], f"Unexpected output shape: {out.shape()}"
    print(f"test_neuron {synap.tensor_data(out)}")


if __name__ == "__main__":
    test_backwardpass()
    print("Tests passed!")
