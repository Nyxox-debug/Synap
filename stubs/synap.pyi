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
        shape:
            A sequence of integers describing the tensor dimensions.
        requires_grad:
            Whether this tensor should track operations for automatic
            differentiation.
        """
        ...

    def clone(self) -> Tensor:
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
            A list of dimension sizes.
        """
        ...

    def view(
        self,
        arg0: collections.abc.Sequence[typing.SupportsInt],
    ) -> Tensor:
        """
        Return a reshaped view of the tensor.

        The total number of elements must remain unchanged.

        Parameters
        ----------
        arg0:
            The new shape.

        Returns
        -------
        Tensor
            A tensor sharing the same underlying data with a new shape.
        """
        ...

    def zero_grad(self) -> None:
        """
        Reset the gradient associated with this tensor to zero.

        This is typically called before a new backward pass.
        """
        ...

    def backward(self, grad_output : Tensor | None = None) -> None:
        """
        Perform backpropagation.

        If grad_output is provided, it is used as the initial gradient.
        Otherwise, the tensor is assumed to be scalar and a gradient of 1
        is used.
        """
        ...

    @property
    def grad(self) -> Tensor:
        """
        The gradient of this tensor with respect to some scalar loss.

        Returns
        -------
        Tensor
            A tensor containing gradient values.
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

    @staticmethod
    def add(a: Tensor, b: Tensor) -> Tensor:
        """
        Add two tensors element-wise.

        Parameters
        ----------
        a:
            First tensor.
        b:
            Second tensor.

        Returns
        -------
        Tensor
            Resulting tensor after addition.
        """
        ...

    @property
    def grad_values(self) -> list[float]:
        """
        The gradient values as a Python list.

        Returns
        -------
        list[float]
            The elements of the grad tensor as a flat list.
        """
        ...

def tensor_data(t: Tensor) -> list[float]:
    """
    Return the tensor values as a Python list.

    Parameters
    ----------
    t:
        Tensor to inspect.

    Returns
    -------
    list[float]
        Elements of the tensor as a flat list.
    """
    ...
