"""Linear surrogate model.

This module defines a simple linear surrogate model for predicting the
log-magnitude of quantum overlaps from truncated one‑hot encoded
Pauli strings.  Each qubit in the input is represented by a length‑3
vector; the identity operator is mapped to an all‑zero vector and the
non‑trivial Paulis (X, Z, Y) are mapped to the canonical basis
vectors.  The flattened representation is passed through a single
fully connected layer to produce a scalar output.  Because the
representation for the identity operator is zero, the linear model
implicitly ignores those positions.
"""

from __future__ import annotations

import torch
from torch import nn


class LinearSurrogate(nn.Module):
    """A simple affine surrogate model for truncated one‑hot inputs.

    This network flattens the one‑hot encoded Pauli string (with three
    channels per qubit) and applies a single linear transformation to
    obtain a scalar prediction.  No hidden layers are used, so this
    model serves as a baseline capturing only additive effects of
    individual qubits.

    Parameters
    ----------
    input_dim: int
        Dimensionality of the flattened input.  For a system with
        ``n_qubits`` qubits, this should be ``n_qubits * 3``.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the linear surrogate.

        Parameters
        ----------
        x: torch.Tensor
            Flattened one‑hot encoded Pauli vectors of shape
            ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Predicted log‑magnitudes of shape ``(batch_size, 1)``.
        """
        return self.linear(x)
