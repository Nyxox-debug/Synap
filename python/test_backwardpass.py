import nn
import synap


def test_backwardpass():
    # Input
    x = synap.Tensor([1, 4], requires_grad=True)
    x.set_values([1, 2, 3, 4])

    # Target
    y = synap.Tensor([1,1], requires_grad=False)
    y.set_values([10])
    print("y-shape: ", y.shape())

    # Model
    m = nn.MLP(4, [4, 4, 4, 4, 1])

    # Forward
    out = m(x)
    print("out requires_grad:", out.requires_grad)

    # Loss (MSE)
    loss = synap.Tensor.mse(out, y)

    # Backward
    loss.backward()

    print("Loss:", synap.tensor_data(loss))

    # Grab a single weight: first layer, first neuron
    w = m.layers[0].neurons[0].w

    print("Weight values:", synap.tensor_data(w))
    print("Gradient values:", w.grad_values)

if __name__ == "__main__":
    test_backwardpass()
