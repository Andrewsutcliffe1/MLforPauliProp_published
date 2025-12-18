"""Graph neural network surrogate.

This module implements a simple message‑passing graph neural network
designed to respect permutation invariance over qubits.  The input
consists of node features for each qubit encoded in a truncated
one‑hot scheme with three channels.  A fixed normalised adjacency
matrix, including self‑loops, encodes the coupling topology of the
underlying Hamiltonian (e.g., a lattice).  The network uses a
sequence of local MLPs followed by message passing steps that
aggregate information from neighbouring qubits.  The final node
representations are averaged to produce a graph representation,
which is then mapped to a scalar prediction.

Unlike many graph neural network implementations, this surrogate
implements the message passing manually using `torch.einsum` to
perform the batched matrix–vector multiplication.  This avoids
external dependencies such as `torch_geometric` and ensures that
aggregation is controlled strictly by the provided adjacency matrix.
"""

from __future__ import annotations

import torch
from torch import nn


class GNNSurrogate(nn.Module):
    """Graph neural network surrogate for Pauli string propagation.

    Parameters
    ----------
    in_features: int, default=3
        Number of channels per qubit.  The truncated one‑hot encoding
        uses three channels: X, Z, Y.  The identity operator is
        represented by an all‑zero vector and therefore does not
        contribute to the learned representation.
    hidden_dim: int, default=32
        Dimensionality of the hidden node representation after the
        lifting layer.
    num_layers: int, default=3
        Number of message‑passing blocks (local MLP + aggregation)
        applied sequentially.
    dropout: float, default=0.1
        Dropout probability used within the local MLPs.
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_dim: int = 32,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # Lifting layer: maps per‑qubit features to hidden space
        self.lift = nn.Linear(in_features, hidden_dim)
        # Construct a module list of local MLPs.  Each MLP processes
        # the hidden node features before message passing.
        mlps = []
        for _ in range(num_layers):
            mlps.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
        self.mlps = nn.ModuleList(mlps)
        # Final readout: average over nodes and map to scalar
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the GNN surrogate.

        Parameters
        ----------
        x: torch.Tensor
            Input node features with shape ``(batch_size, n_qubits, in_features)``.
        adj: torch.Tensor
            Normalised adjacency matrix with self‑loops of shape
            ``(n_qubits, n_qubits)``.  The matrix should be row
            normalised such that each row sums to one.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch_size, 1)`` containing the
            predicted log‑magnitudes.
        """
        # Lift the input features into the hidden space
        h = self.lift(x)  # (batch, n_qubits, hidden_dim)
        # Apply a series of message‑passing blocks
        for mlp in self.mlps:
            # Local transformation
            local = mlp(h)  # (batch, n_qubits, hidden_dim)
            # Aggregate neighbour information using the fixed adjacency
            # matrix.  Use einsum to perform batched matrix–vector
            # multiplication across the node dimension.  adj has
            # shape (n_qubits, n_qubits).  local has shape
            # (batch, n_qubits, hidden_dim).  The resulting h has
            # shape (batch, n_qubits, hidden_dim).
            h = torch.einsum('ij,bjk->bik', adj, local)
        # Pool across nodes to obtain a graph representation per sample
        graph_repr = h.mean(dim=1)  # (batch, hidden_dim)
        # Map the pooled representation to a scalar
        out = self.readout(graph_repr)  # (batch, 1)
        return out
