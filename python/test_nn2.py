import nn
import synap

def test_neuron():
    x = synap.Tensor([2,4], requires_grad=True)
    x.set_values([1, 2, 3, 4, 5, 6, 7, 8])
    n = nn.Neuron(4)
    n(x)

def test_layer():
    x = synap.Tensor([2,2], requires_grad=True)
    x.set_values([1, 2, 3, 4])
    lyr = nn.Layer(nin=2, nout=4)
    out = lyr(x)
    synap.tensor_data(out)

def test_MLP():
    print()

if __name__ == "__main__":
    test_MLP()
