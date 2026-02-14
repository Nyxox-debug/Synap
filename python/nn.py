import random
import synap

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        # Weight tensor: shape [nin, 1], requires_grad=True
        self.w = synap.Tensor([nin, 1], requires_grad=True)
        self.w.set_values([random.uniform(-1,1) for _ in range(nin)])
        # Bias tensor: shape [1], requires_grad=True
        self.b = synap.Tensor([1], requires_grad=True)
        self.b.set_values([0])
        self.nonlin = nonlin

    def __call__(self, x: synap.Tensor):
        # x: Tensor of shape [nin]
        if len(x.shape()) == 1:
            x = x.view([x.shape()[0], 1])  # ensure column vector
        print(synap.tensor_data(x))
        z = synap.Tensor.add(synap.Tensor.matmul(x, self.w), self.b)
        print(synap.tensor_data(z))
        print(8*"-")
        if self.nonlin:
            z = synap.Tensor.relu(z)
        return z

        def parameters(self):
            return [self.w, self.b]

        def __repr__(self):
            return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({self.w.shape()[0]})"

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: synap.Tensor):
        outputs = [n(x) for n in self.neurons]
        if len(outputs) == 1:
            return outputs[0]
        # concatenate outputs along last axis
        out_values = []
        for o in outputs:
            out_values.extend(synap.tensor_data(o))
        out = synap.Tensor([len(out_values)], requires_grad=True)
        out.set_values(out_values)
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i+1], nonlin=(i != len(nouts)-1))
            for i in range(len(nouts))
        ]

    def __call__(self, x: synap.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
