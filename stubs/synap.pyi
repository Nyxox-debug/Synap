"""Synap — a mini deep learning framework with Tensors and autodiff."""
from __future__ import annotations
import collections.abc
import typing

__all__: list[str] = ["Tensor"]

class Tensor:
    """
    A multi-dimensional array object with optional automatic differentiation.

    `Tensor` represents the core data structure of the system. It stores shape
    information and participates in gradient-based computation when
    `requires_grad` is enabled.
    """

    def __init__(
        self,
        shape: collections.abc.Sequence[typing.SupportsInt],
        requires_grad: bool = False,
    ) -> None:
        """
        Create a new Tensor.

        Parameters
        ----------
        shape : Sequence[int]
            The dimensions of the tensor.
        requires_grad : bool, optional
            Whether to track operations for automatic differentiation (default False).
        """
        ...

    def clone(self) -> "Tensor":
        """
        Return a copy of this tensor.

        The cloned tensor has the same shape and data, but is a distinct object.
        Gradient tracking behavior is preserved.
        """
        ...

    def shape(self) -> list[int]:
        """
        Return the shape of the tensor.

        Returns
        -------
        list[int]
            List of dimension sizes.
        """
        ...

    def view(self, new_shape: collections.abc.Sequence[typing.SupportsInt]) -> "Tensor":
        """
        Return a reshaped view of the tensor.

        Parameters
        ----------
        new_shape : Sequence[int]
            New shape for the tensor. Must preserve the total number of elements.

        Returns
        -------
        Tensor
            A tensor sharing the same underlying data with the new shape.
        """
        ...

    def zero_grad(self) -> None:
        """
        Reset the gradient associated with this tensor to zero.
        """
        ...

    def backward(self, grad_output: "Tensor | None" = None) -> None:
        """
        Perform backpropagation starting from this tensor.

        Parameters
        ----------
        grad_output : Tensor or None
            The initial upstream gradient. If None, assumes a scalar sink
            and uses ones as the gradient.
        """
        ...

    @property
    def grad(self) -> "Tensor":
        """
        The gradient of this tensor with respect to some scalar loss.

        Returns
        -------
        Tensor
            Tensor containing gradient values.
        """
        ...

    @property
    def requires_grad(self) -> bool:
        """
        Whether this tensor tracks gradients.

        Returns
        -------
        bool
            True if gradient tracking is enabled.
        """
        ...

    @property
    def grad_values(self) -> list[float]:
        """
        Return the gradient values as a flat Python list.

        Returns
        -------
        list[float]
            Elements of the grad tensor as a flat list.
        """
        ...

    @staticmethod
    def add(a: "Tensor", b: "Tensor") -> "Tensor":
        """
        Element-wise addition of two tensors.

        Parameters
        ----------
        a : Tensor
            First tensor.
        b : Tensor
            Second tensor.

        Returns
        -------
        Tensor
            Result of adding a and b element-wise.
        """
        ...

    @staticmethod
    def sub(a: "Tensor", b: "Tensor") -> "Tensor":
        """
        Element-wise subtraction of two tensors.

        Parameters
        ----------
        a : Tensor
            Minuend tensor.
        b : Tensor
            Subtrahend tensor.

        Returns
        -------
        Tensor
            Result of subtracting b from a element-wise.
        """
        ...

    @staticmethod
    def mul(a: "Tensor", b: "Tensor") -> "Tensor":
        """
        Element-wise multiplication of two tensors.

        Parameters
        ----------
        a : Tensor
            First tensor.
        b : Tensor
            Second tensor.

        Returns
        -------
        Tensor
            Result of multiplying a and b element-wise.
        """
        ...

    @staticmethod
    def div(a: "Tensor", b: "Tensor") -> "Tensor":
        """
        Element-wise division of two tensors.

        Parameters
        ----------
        a : Tensor
            Dividend tensor.
        b : Tensor
            Divisor tensor.

        Returns
        -------
        Tensor
            Result of dividing a by b element-wise.
        """
        ...
    @staticmethod
    def mean(a: "Tensor") -> "Tensor":
        """
        Compute the arithmetic mean of all elements in a tensor.

        Acts as a scalar sink for backpropagation, reducing the tensor
        to a single scalar value.

        Parameters
        ----------
        a : Tensor
            Input tensor of any shape.

        Returns
        -------
        Tensor
            Scalar tensor containing the average of all elements in `a`.
        """
        ...

    @staticmethod
    def sum(a: "Tensor") -> "Tensor":
        """
        Reduce a tensor to a scalar by summing all elements.

        Acts as a scalar sink for backpropagation.

        Parameters
        ----------
        a : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar tensor containing the sum of all elements.
        """
        ...

    def set_values(self, values: list[float]) -> None:
        """
        Set the elements of the tensor to the given values.

        Parameters
        ----------
        values : list[float]
            Flattened list of values to assign to the tensor. Must match
            the number of elements in the tensor.
        """
        ...

    @staticmethod
    def transpose(a: "Tensor") -> "Tensor":
        """
        Return the transpose of a 2D tensor.

        Forward pass swaps rows and columns. Used in matmul and other linear
        algebra operations. Backward pass transposes the gradient.

        Parameters
        ----------
        a : Tensor
            Input 2D tensor.

        Returns
        -------
        Tensor
            New tensor containing the transposed data.
        """
        ...

    @staticmethod
    def matmul(a: "Tensor", b: "Tensor") -> "Tensor":
        """
        Perform matrix multiplication of two 2D tensors.

        Forward pass computes standard matrix product. Backward pass propagates
        gradients to both inputs using:
            dA = grad_out @ B.T
            dB = A.T @ grad_out

        Parameters
        ----------
        a : Tensor
            Left-hand side 2D tensor.
        b : Tensor
            Right-hand side 2D tensor.

        Returns
        -------
        Tensor
            Result of matrix multiplication.
        """
        ...

    @staticmethod
    def relu(a: "Tensor") -> "Tensor":
        """
        Apply the ReLU (Rectified Linear Unit) activation element-wise.

        Forward pass: output = max(0, input)
        Backward pass: gradient is propagated only for positive input elements.

        Parameters
        ----------
        a : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Tensor with ReLU applied element-wise.
        """
        ...

    @staticmethod
    def sigmoid(a: "Tensor") -> "Tensor":
        """
        Apply the Sigmoid activation element-wise.

        Forward pass: output = 1 / (1 + exp(-input))
        Backward pass: gradient = output * (1 - output)

        Parameters
        ----------
        a : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Tensor with Sigmoid applied element-wise.
        """
        ...

    @staticmethod
    def tanh(a: "Tensor") -> "Tensor":
        """
        Apply the Tanh activation element-wise.

        Forward pass: output = tanh(input)
        Backward pass: gradient = 1 - output^2

        Parameters
        ----------
        a : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Tensor with Tanh applied element-wise.
        """
        ...

    @staticmethod
    def mse(pred: "Tensor", target: "Tensor") -> "Tensor":
        """
        Compute the Mean Squared Error (MSE) between predictions and targets.

        Forward pass: MSE = mean((pred - target)^2)
        Backward pass: dL/dpred = 2*(pred - target)/n

        Parameters
        ----------
        pred : Tensor
            Predicted values.
        target : Tensor
            Ground-truth values.

        Returns
        -------
        Tensor
            Scalar tensor containing the mean squared error.
        """
        ...

    @staticmethod
    def softmax_cross_entropy(logits: "Tensor", targets: "Tensor") -> "Tensor":
        """
        Compute the Softmax Cross-Entropy loss.

        Forward pass:
            1. Shift logits for numerical stability: logits_shifted = logits - max(logits, axis=1)
            2. Compute softmax probabilities: probs = exp(logits_shifted) / sum(exp(logits_shifted), axis=1)
            3. Compute cross-entropy per row: -sum(targets * log(probs), axis=1)
            4. Take mean over batch

        Backward pass:
            Gradient w.r.t logits: (probs - targets) / batch_size

        Parameters
        ----------
        logits : Tensor
            Raw prediction scores (pre-softmax).
        targets : Tensor
            One-hot encoded target labels.

        Returns
        -------
        Tensor
            Scalar tensor containing the mean cross-entropy loss over the batch.
        """
        ...
def tensor_data(t: Tensor) -> list[float]:
    """
    Return the tensor values as a Python list.

    Parameters
    ----------
    t : Tensor
        Tensor to inspect.

    Returns
    -------
    list[float]
        Elements of the tensor as a flat list.
    """
    ...
