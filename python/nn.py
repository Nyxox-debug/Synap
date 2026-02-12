import synap
import random


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    def __init__(self, in_features, out_features):
        # Xavier-style uniform init (simple version)
        limit = (1.0 / in_features) ** 0.5

        self.W = synap.Tensor([in_features, out_features], requires_grad=True)
        self.b = synap.Tensor([out_features], requires_grad=True)

        w_vals = [random.uniform(-limit, limit) for _ in range(in_features * out_features)]
        b_vals = [0.0 for _ in range(out_features)]

        self.W.set_values(w_vals)
        self.b.set_values(b_vals)

    def __call__(self, x):
        out = synap.Tensor.matmul(x, self.W)
        out = synap.Tensor.add(out, self.b)
        return out

    def parameters(self):
        return [self.W, self.b]

    def __repr__(self):
        return f"Linear({self.W.shape()[0]}, {self.W.shape()[1]})"


class ReLU(Module):
    def __call__(self, x):
        return synap.Tensor.relu(x)


class Sigmoid(Module):
    def __call__(self, x):
        return synap.Tensor.sigmoid(x)


class Tanh(Module):
    def __call__(self, x):
        return synap.Tensor.tanh(x)


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class MLP(Module):
    def __init__(self, sizes, activation="relu"):
        self.layers = []

        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1]))

            if i != len(sizes) - 2:
                if activation == "relu":
                    self.layers.append(ReLU())
                elif activation == "sigmoid":
                    self.layers.append(Sigmoid())
                elif activation == "tanh":
                    self.layers.append(Tanh())

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP({self.layers})"


# Simple SGD optimizer
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue

            grad_vals = p.grad_values
            data_vals = synap.tensor_data(p)

            updated = [
                d - self.lr * g
                for d, g in zip(data_vals, grad_vals)
            ]

            p.set_values(updated)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()
