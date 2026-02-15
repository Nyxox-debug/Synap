import nn
import synap

def test_forwardpass():
    x = synap.Tensor([1, 4], requires_grad=True)
    x.set_values([1, 2, 3, 4])
    m = nn.MLP(4, [4, 4, 1])
    out = m(x)

    assert isinstance(out, synap.Tensor), "Neuron output should be a Tensor"
    assert out.shape() == [1], f"Unexpected output shape: {out.shape()}"
    print(f"test_neuron {synap.tensor_data(out)}")


if __name__ == "__main__":
    test_forwardpass()
    print("Tests passed!")
