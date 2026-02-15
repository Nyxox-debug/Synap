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
    x = synap.Tensor([2,2], requires_grad=True)
    x.set_values([1, 2, 3, 4])
    x = x.view([1, 4])
    mlp = nn.MLP(4, [8, 8, 4, 1])
    out = mlp(x)

    print(mlp)
    print("Output:", synap.tensor_data(out))

def test_MLP_5x5():
    # 5x5 grayscale image
    x = synap.Tensor([5, 5], requires_grad=True)

    # Example pixel values (25 values)
    x.set_values([
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.5, 0.4, 0.3, 0.2, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.9, 0.8, 0.7, 0.6, 0.5,
        0.0, 0.2, 0.4, 0.6, 0.8
    ])

    # Flatten 5x5 → 25
    x = x.view([1,25])

    # MLP: 25 inputs
    mlp = nn.MLP(25, [16, 8, 1])

    out = mlp(x)

    print(mlp)
    print("Output:", synap.tensor_data(out))

if __name__ == "__main__":
    test_MLP()
