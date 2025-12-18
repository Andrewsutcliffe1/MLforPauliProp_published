"""Quadratic surrogate model.

This module implements a surrogate that includes both linear and
quadratic interactions between qubit features.  The input is a
flattened truncated one‑hot encoding of Pauli strings (three
channels per qubit).  The model computes

.. math::

    f(x) = b + \mathbf{w}^\top x + x^\top Q x,

where :math:`\mathbf{w}` is a learnable weight vector, :math:`b` is a
bias term (embedded in the linear layer) and :math:`Q` is a learnable
matrix capturing pairwise interactions.  The quadratic term is
implemented efficiently via a tensor contraction to avoid explicit
looping over the batch dimension.
"""

from __future__ import annotations

import torch
from torch import nn


class QuadraticSurrogate(nn.Module):
    """Surrogate model with linear and quadratic components.

    Parameters
    ----------
    input_dim: int
        Dimensionality of the flattened truncated one‑hot input.  For
        ``n_qubits`` qubits, this should be ``n_qubits * 3``.

    Notes
    -----
    The quadratic weight matrix :math:`Q` has shape
    ``(input_dim, input_dim)`` and is initialised with small values
    drawn from a normal distribution.  The bias is incorporated into
    the linear component.  Because the identity Pauli is mapped to an
    all‑zero vector, interactions involving the identity operator have
    no effect on the output.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # Linear component including bias
        self.linear = nn.Linear(input_dim, 1)
        # Quadratic term: learnable symmetric matrix
        # We do not enforce symmetry explicitly but allow optimisation
        # to learn the appropriate structure.  Initialising with small
        # values helps stabilise early training.
        self.Q = nn.Parameter(torch.randn(input_dim, input_dim) * 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the linear + quadratic prediction.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch_size, 1)`` with the
            predicted log‑magnitudes.
        """
        # Linear contribution (includes bias)
        linear_term = self.linear(x)  # (batch_size, 1)
        # Quadratic contribution: x^T Q x per sample
        # Use einsum for batched bilinear form
        quad_term = torch.einsum('bi,ij,bj->b', x, self.Q, x).unsqueeze(-1)
        return linear_term + quad_term
