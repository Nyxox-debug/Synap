"""Synap — a mini deep learning framework with Tensors and autodiff."""
from __future__ import annotations
import collections.abc
import typing

__all__: list[str] = ["Tensor", "Value"]


class Value:
    """
    Scalar value supporting reverse-mode automatic differentiation.

    `Value` represents a single scalar node in a computational graph.
    It stores a numerical value, accumulates gradients, and supports
    backpropagation via `backward()`.
    """

    data: float
    grad: float

    def __init__(self, data: float) -> None:
        """
        Create a scalar value.

        Parameters
        ----------
        data:
            The numerical value of this node.
        """
        ...

    def backward(self) -> None:
        """
        Perform reverse-mode automatic differentiation.

        Computes gradients for all ancestor nodes in the computation graph.
        """
        ...

    def __add__(self, other: Value) -> Value:
        ...

    def __mul__(self, other: Value) -> Value:
        ...


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
