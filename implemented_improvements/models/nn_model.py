"""Fully connected neural network surrogate.

This module defines a shallow feedforward neural network that maps
flattened truncated one‑hot Pauli encodings to scalar predictions.
The network comprises several dense layers with ReLU activations in
between.  It is intentionally kept shallow to balance capacity and
trainability on relatively modest datasets.
"""

from __future__ import annotations

import torch
from torch import nn


class NNSurrogate(nn.Module):
    """A shallow fully connected neural network surrogate.

    Parameters
    ----------
    input_dim: int
        Dimensionality of the flattened truncated one‑hot input.  For
        ``n_qubits`` qubits this is ``n_qubits * 3``.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Parameters
        ----------
        x: torch.Tensor
            Flattened truncated one‑hot encoded Pauli strings of shape
            ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch_size, 1)`` representing
            predicted log‑magnitudes.
        """
        return self.net(x)
