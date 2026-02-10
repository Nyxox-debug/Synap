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
