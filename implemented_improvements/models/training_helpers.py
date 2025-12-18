"""Training utilities for surrogate models.

This module provides helper classes and functions used across the
surrogate training pipeline.  It includes tools for constructing
graph topologies, converting integer Pauli encodings into truncated
one‑hot tensors, splitting datasets into train/validation/test
partitions, and computing weighted loss functions.  Wherever
possible, functionality is packaged as pure functions or simple
PyTorch datasets so that training scripts remain concise and easy to
understand.
"""

from __future__ import annotations

import math
import re
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split

Edge = Tuple[int, int]

# -----------------------------------------------------------------------------
# Graph construction helpers
# -----------------------------------------------------------------------------

def ring_topology(n: int) -> List[Edge]:
    """Create a ring (cycle) topology on ``n`` nodes.

    The nodes are numbered from 1 to ``n`` inclusive and edges are
    undirected.  Self‑loops are not included here; they are added
    later when constructing the adjacency matrix.

    Parameters
    ----------
    n: int
        Number of nodes in the ring.  Must be at least 3.

    Returns
    -------
    List[Edge]
        List of edges forming the ring.
    """
    if n < 3:
        raise ValueError("ring_topology requires n >= 3")
    edges: List[Edge] = []
    for i in range(1, n):
        edges.append((i, i + 1))
    edges.append((n, 1))
    return edges


def rect_topology(r: int, c: int) -> List[Edge]:
    """Create a rectangular grid topology of size ``r`` by ``c``.

    Nodes are numbered row‑major from 1 to ``r*c``.  Each node
    connects to its right and bottom neighbours (if they exist).

    Parameters
    ----------
    r: int
        Number of rows in the grid.
    c: int
        Number of columns in the grid.

    Returns
    -------
    List[Edge]
        List of edges representing nearest‑neighbour couplings.
    """
    if r < 1 or c < 1:
        raise ValueError("rect_topology requires r,c >= 1")
    edges: List[Edge] = []
    # Helper to convert (row, col) into a 1-indexed node ID
    def idx(rr: int, cc: int) -> int:
        return rr * c + cc + 1
    for rr in range(r):
        for cc in range(c):
            u = idx(rr, cc)
            if cc + 1 < c:
                edges.append((u, idx(rr, cc + 1)))
            if rr + 1 < r:
                edges.append((u, idx(rr + 1, cc)))
    return edges


def _edges_to_normalized_adjacency(
    edges: List[Edge],
    n_qubits: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    add_self_loops: bool = True,
    row_normalize: bool = True,
) -> torch.Tensor:
    """Convert an edge list into a normalised adjacency matrix.

    This helper accepts either 0‑indexed or 1‑indexed edge lists and
    produces a square adjacency matrix of shape ``(n_qubits, n_qubits)``.
    Optionally, self‑loops can be added and each row can be normalised
    so that it sums to one.  The resulting matrix is suitable for
    deterministic message passing in the GNN surrogate.

    Parameters
    ----------
    edges: List[Edge]
        List of edges connecting qubits.  Each edge is a tuple
        ``(u, v)``.  Indexing can be 0‑based or 1‑based and will be
        auto‑detected.
    n_qubits: int
        Total number of qubits (nodes) in the system.
    device: torch.device, optional
        Device on which to allocate the adjacency tensor.
    dtype: torch.dtype, default=torch.float32
        Data type of the adjacency tensor.
    add_self_loops: bool, default=True
        If ``True``, add self‑loops to every node.
    row_normalize: bool, default=True
        If ``True``, normalise rows so that they sum to one.

    Returns
    -------
    torch.Tensor
        Normalised adjacency matrix of shape ``(n_qubits, n_qubits)``.
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    # Handle trivial case (no edges)
    if len(edges) == 0:
        adj = torch.zeros((n_qubits, n_qubits), dtype=dtype, device=device)
        if add_self_loops:
            adj = adj + torch.eye(n_qubits, dtype=dtype, device=device)
        if row_normalize:
            deg = adj.sum(dim=1).clamp(min=1.0)
            adj = adj / deg.unsqueeze(1)
        return adj
    # Determine whether indexing is 0‑based or 1‑based
    mins = min(min(u, v) for (u, v) in edges)
    maxs = max(max(u, v) for (u, v) in edges)
    one_indexed = (mins >= 1) and (maxs <= n_qubits)
    zero_indexed = (mins >= 0) and (maxs < n_qubits)
    if one_indexed and not zero_indexed:
        edges0 = [(u - 1, v - 1) for (u, v) in edges]
    elif zero_indexed and not one_indexed:
        edges0 = edges
    elif one_indexed and zero_indexed:
        # Ambiguous indexing: if any node equals n_qubits assume 1‑indexing
        if any(u == n_qubits or v == n_qubits for (u, v) in edges):
            edges0 = [(u - 1, v - 1) for (u, v) in edges]
        else:
            edges0 = edges
    else:
        raise ValueError(f"Edges out of range: min={mins}, max={maxs}, n={n_qubits}")
    # Construct adjacency
    adj = torch.zeros((n_qubits, n_qubits), dtype=dtype, device=device)
    for u, v in edges0:
        if u < 0 or v < 0 or u >= n_qubits or v >= n_qubits:
            raise ValueError(f"Edge ({u},{v}) out of bounds for n={n_qubits}")
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    if add_self_loops:
        adj = adj + torch.eye(n_qubits, dtype=dtype, device=device)
    if row_normalize:
        deg = adj.sum(dim=1).clamp(min=1.0)
        adj = adj / deg.unsqueeze(1)
    return adj


def topology_from_name(
    name: str,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[int, torch.Tensor]:
    """Parse a canonical dataset name to produce its topology.

    Supported naming conventions are:

    * ``"ring_<n>"`` — a 1D ring of ``n`` qubits.
    * ``"rect_<r>x<c>"`` — an ``r`` by ``c`` rectangular grid.

    The function returns both the number of qubits and the
    normalised adjacency matrix.  See the documentation for
    :func:`_edges_to_normalized_adjacency` for details on the matrix.

    Parameters
    ----------
    name: str
        Dataset name encoding the topology.
    device: torch.device, optional
        Device on which to place the adjacency matrix.
    dtype: torch.dtype, default=torch.float32
        Data type of the adjacency matrix.

    Returns
    -------
    Tuple[int, torch.Tensor]
        A tuple ``(n_qubits, adj)`` where ``adj`` is the normalised
        adjacency matrix.
    """
    name = name.strip().lower()
    m = re.fullmatch(r"ring_(\d+)", name)
    if m:
        n = int(m.group(1))
        edges = ring_topology(n)
        return n, _edges_to_normalized_adjacency(edges, n, device=device, dtype=dtype)
    m = re.fullmatch(r"rect_(\d+)x(\d+)", name)
    if m:
        r, c = int(m.group(1)), int(m.group(2))
        n = r * c
        edges = rect_topology(r, c)
        return n, _edges_to_normalized_adjacency(edges, n, device=device, dtype=dtype)
    raise ValueError(
        f"Unrecognised topology name '{name}'. Expected 'ring_<n>' or 'rect_<r>x<c>'."
    )


def get_adjacency_matrix(
    topology_list: List[Tuple[int, int]],
    n_qubits: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert an explicit edge list to a normalised adjacency matrix.

    This function is provided for backwards compatibility and simply
    calls :func:`_edges_to_normalized_adjacency`.  It supports both
    0‑based and 1‑based indexing schemes.

    Parameters
    ----------
    topology_list: List[Edge]
        List of edges connecting qubits.
    n_qubits: int
        Total number of qubits.
    device: torch.device, optional
        Device on which to allocate the adjacency tensor.
    dtype: torch.dtype, default=torch.float32
        Data type of the adjacency tensor.

    Returns
    -------
    torch.Tensor
        Normalised adjacency matrix of shape ``(n_qubits, n_qubits)``.
    """
    return _edges_to_normalized_adjacency(
        list(topology_list),
        n_qubits,
        device=device,
        dtype=dtype,
        add_self_loops=True,
        row_normalize=True,
    )


# -----------------------------------------------------------------------------
# Dataset and loss
# -----------------------------------------------------------------------------

class PauliDataset(Dataset):
    """PyTorch dataset representing truncated Pauli strings.

    Each entry in the dataset corresponds to a Pauli string encoded
    as an integer and a target complex amplitude ``f1``.  The integer
    encoding is interpreted as a base‑4 number, with the least
    significant digit corresponding to the last qubit.  However,
    because the identity operator is treated as the zero vector, each
    digit 0 maps to the zero vector instead of a one‑hot indicator.

    The target values are converted to their logarithmic magnitude
    with a configurable small ``epsilon`` added (or clamped) to ensure
    numerical stability.  By transforming the targets once at dataset
    construction time, downstream training code can treat the problem
    as a standard regression task in log‑space without having to
    perform any logarithms in the loss function.  Optional sample
    weights (``orbit_size``) are supported for symmetry reduction.  If
    no weight is provided in the CSV, it defaults to one.

    Parameters
    ----------
    csv_path: str
        Path to the CSV file containing columns ``observable``, ``f1``
        and optionally ``orbit_size``.
    n_qubits: int, optional
        Explicit number of qubits to assume.  If not provided, it is
        inferred from the maximum observable value.
    epsilon: float, default=1e-12
        Minimum positive value used when computing the logarithm of
        the target magnitude.  Target magnitudes smaller than
        ``epsilon`` are clamped to ``epsilon`` before taking the
        logarithm.  This avoids computing ``log(0)`` which would
        otherwise produce ``-inf``.
    """

    def __init__(
        self,
        csv_path: str,
        n_qubits: Optional[int] = None,
        *,
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)
        # Infer number of qubits from the maximum observable if not provided
        if n_qubits is None:
            max_obs = int(self.df["observable"].max())
            if max_obs == 0:
                inferred = 1
            else:
                # Determine number of base‑4 digits required to represent max_obs
                inferred = int(math.ceil(math.log(max_obs + 1e-9 + 1, 4)))
            self.n_qubits = inferred
        else:
            self.n_qubits = int(n_qubits)
        # Store epsilon for numerical stability when computing log magnitudes
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon: float = float(epsilon)
        self.x_data: List[torch.Tensor] = []
        self.y_data: List[torch.Tensor] = []
        self.w_data: List[torch.Tensor] = []
        for _, row in self.df.iterrows():
            obs = int(row["observable"])
            f1_val = row["f1"]
            # Convert the target to a complex number if needed
            try:
                f1_c = complex(f1_val)
            except Exception:
                f1_c = float(f1_val)
            magnitude = abs(f1_c)
            # Clamp the magnitude to epsilon to avoid log(0) or extremely large negatives
            mag_clamped = magnitude if magnitude >= self.epsilon else self.epsilon
            log_mag = math.log10(mag_clamped)
            one_hot = self._int_to_truncated_one_hot(obs)
            weight = row.get("orbit_size", 1.0)
            # Flatten weight to a scalar tensor so that downstream loss functions can broadcast correctly
            self.x_data.append(one_hot)
            self.y_data.append(torch.tensor([log_mag], dtype=torch.float32))
            self.w_data.append(torch.tensor([float(weight)], dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.x_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.x_data[idx],
            "y": self.y_data[idx],
            "w": self.w_data[idx],
        }

    def _int_to_truncated_one_hot(self, value: int) -> torch.Tensor:
        """Convert an integer to a truncated one‑hot tensor.

        The integer ``value`` is interpreted as a base‑4 number.  Each
        digit corresponds to a Pauli operator: 0 → identity, 1 → X,
        2 → Z, 3 → Y.  The identity is encoded as a zero vector and
        the other operators are encoded as the standard basis vectors
        in ``R^3``.  The qubit ordering is big‑endian: the most
        significant digit is placed at index 0.

        Parameters
        ----------
        value: int
            Integer encoding of the Pauli string.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(n_qubits, 3)`` representing the truncated
            one‑hot encoding.
        """
        digits = []
        num = int(value)
        for _ in range(self.n_qubits):
            digits.append(num % 4)
            num //= 4
        # Reverse to make index 0 correspond to the most significant qubit
        digits = digits[::-1]
        # Allocate zero tensor for the one‑hot representation
        one_hot = torch.zeros(self.n_qubits, 3, dtype=torch.float32)
        for qi, d in enumerate(digits):
            if d == 0:
                # Identity: leave the row as zeros
                continue
            elif d == 1:
                one_hot[qi, 0] = 1.0  # X
            elif d == 2:
                one_hot[qi, 1] = 1.0  # Z
            elif d == 3:
                one_hot[qi, 2] = 1.0  # Y
            else:
                raise ValueError(f"Invalid base‑4 digit {d} for Pauli encoding")
        return one_hot


def make_train_val_test_loaders(
    dataset: Dataset,
    *,
    batch_size: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split a dataset into train, validation and test loaders.

    The dataset is partitioned randomly according to the specified
    fractions.  The splits are non‑overlapping and their sizes sum
    exactly to the length of the dataset.  The loaders share the same
    batch size.  If desired, the training loader can shuffle the
    samples each epoch.

    Parameters
    ----------
    dataset: Dataset
        The full dataset to split.
    batch_size: int
        Mini‑batch size for all loaders.
    train_frac: float, default=0.8
        Fraction of data to allocate to the training set.
    val_frac: float, default=0.1
        Fraction of data to allocate to the validation set.  The test
        fraction is implicitly ``1 - train_frac - val_frac``.
    seed: int, default=0
        Seed controlling the randomness of the split.
    shuffle_train: bool, default=True
        If ``True``, shuffle the training loader each epoch.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        The training, validation and test loaders respectively.
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")
    N = len(dataset)
    n_train = int(train_frac * N)
    n_val = int(val_frac * N)
    n_test = N - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader

def make_train_test_loaders(
    dataset,
    *,
    batch_size: int,
    train_frac: float = 0.8,
    seed: int = 0,
    shuffle_train: bool = True,
):
    """
    Convenience wrapper around make_train_val_test_loaders
    for the no-validation case.
    """
    train_loader, _, test_loader = make_train_val_test_loaders(
        dataset,
        batch_size=batch_size,
        train_frac=train_frac,
        val_frac=0.0,
        seed=seed,
        shuffle_train=shuffle_train,
    )
    return train_loader, test_loader


def orbit_weighted_mse(
    preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Compute the orbit‑weighted mean squared error.

    The error is defined as:

    .. math::

        \mathrm{MSE} = \frac{\sum_i w_i (p_i - t_i)^2}{\sum_i w_i},

    where :math:`p_i` are the predictions, :math:`t_i` are the targets
    and :math:`w_i` are the orbit weights.  A small epsilon is used
    when dividing by the total weight to avoid division by zero.

    Parameters
    ----------
    preds: torch.Tensor
        Predicted values of shape ``(batch_size, 1)``.
    targets: torch.Tensor
        Target values of shape ``(batch_size, 1)``.
    weights: torch.Tensor
        Sample weights of shape ``(batch_size, 1)``.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the weighted mean squared error.
    """
    diff = preds.squeeze() - targets.squeeze()
    weighted_sq = weights.squeeze() * diff.pow(2)
    total_weight = weights.squeeze().sum().clamp(min=1.0)
    return weighted_sq.sum() / total_weight


@torch.no_grad()
def fit_linear_closed_form(train_loader, *, device, ridge_lambda: float = 0.0):
    """
    Weighted least squares (optionally ridge) fit for y = X w + b.
    Returns (w, b) tensors on `device`.
    """
    Xs, ys, ws = [], [], []
    for batch in train_loader:
        x = batch["x"].to(device).view(batch["x"].size(0), -1).float()
        y = batch["y"].to(device).view(-1).float()
        w = batch["w"].to(device).view(-1).float()
        Xs.append(x); ys.append(y); ws.append(w)

    X = torch.cat(Xs, dim=0)          # (N, d)
    y = torch.cat(ys, dim=0)          # (N,)
    w = torch.cat(ws, dim=0)          # (N,)

    N, d = X.shape

    # Add bias column
    ones = torch.ones((N, 1), device=device, dtype=X.dtype)
    Xb = torch.cat([X, ones], dim=1)  # (N, d+1)

    # Apply weights via sqrt(w): solve min || sqrt(W)(Xb theta - y) ||^2
    sw = torch.sqrt(torch.clamp(w, min=0.0)).unsqueeze(1)  # (N,1)
    Xw = Xb * sw
    yw = y * sw.squeeze(1)

    # Normal equations
    A = Xw.T @ Xw                      # (d+1, d+1)
    b = Xw.T @ yw                      # (d+1,)

    if ridge_lambda > 0:
        R = torch.eye(d + 1, device=device, dtype=X.dtype)
        R[-1, -1] = 0.0               # don't regularize bias
        A = A + ridge_lambda * R

    theta = torch.linalg.solve(A, b)   # (d+1,)

    w_hat = theta[:-1].contiguous()
    b_hat = theta[-1].contiguous()
    return w_hat, b_hat

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
import matplotlib as mpl

mpl.rcParams.update({

    # --- Figure size & resolution ---
    "figure.figsize": (3.0, 3.0),     # larger for log plots
    "figure.dpi": 180,
    "savefig.dpi": 300,

    # --- Fonts ---
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",

    # --- Font sizes ---
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,

    # --- Axes & ticks ---
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,

    # --- Grid (off by default for physics plots) ---
    "axes.grid": True,

    # --- Lines ---
    "lines.linewidth": 1.5,
    "lines.markersize": 4,

    # --- Legend ---
    "legend.frameon": False,

    # --- LaTeX-like layout ---
    "axes.unicode_minus": False,
    "text.usetex": False,  # keep False for portability
})

def parity_plot_train_test(
    model,
    train_loader,
    test_loader,
    *,
    device,
    c_trunc: float = 1e-7,   # linear space threshold ALWAYS
    log_base: int = 10,      # assumes your y-values are log10 by default
    alpha=0.6,
    s=12,
):
    """
    Parity plot (true vs predicted) for train and test sets + truncation threshold.

    - c_trunc is ALWAYS specified in linear space (e.g. 1e-7).
    - y-values (t(P), M(P)) are assumed to be in log-space (log10 by default).

    Confusion convention: "positive = TRUNCATE"
        truncate if value < log(c_trunc)
        keep     if value >= log(c_trunc)
    """

    model.eval()

    def collect(loader):
        y_true, y_pred = [], []
        for batch in loader:
            x = batch["x"].to(device).view(batch["x"].size(0), -1)
            y = batch["y"].to(device)

            with torch.no_grad():
                pred = model(x)

            y_true.append(y.detach().cpu())
            y_pred.append(pred.detach().cpu())

        return (
            torch.cat(y_true).numpy().ravel(),
            torch.cat(y_pred).numpy().ravel(),
        )

    y_train, yhat_train = collect(train_loader)
    y_test,  yhat_test  = collect(test_loader)

    # --- convert linear threshold -> log-space line ---
    if c_trunc <= 0:
        raise ValueError("c_trunc must be > 0 (linear space threshold).")

    if log_base == 10:
        c_line = math.log10(float(c_trunc))
    elif log_base in ("e", math.e):
        c_line = math.log(float(c_trunc))
    else:
        raise ValueError("log_base must be 10 or 'e' (math.e).")

    # --- confusion counts (positive = truncate, i.e. value < c_line) ---
    def confusion(y, yhat):
        c = c_line
        tp = ((y <  c) & (yhat <  c)).sum()   # correctly truncated
        tn = ((y >= c) & (yhat >= c)).sum()   # correctly kept
        fp = ((y >= c) & (yhat <  c)).sum()   # wrongly truncated
        fn = ((y <  c) & (yhat >= c)).sum()   # missed truncation
        return int(tp), int(fp), int(fn), int(tn)

    tp_tr, fp_tr, fn_tr, tn_tr = confusion(y_train, yhat_train)
    tp_te, fp_te, fn_te, tn_te = confusion(y_test,  yhat_test)

    def rates(tp, fp, fn, tn):
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) else float("nan")
        fpr  = fp / (fp + tn) if (fp + tn) else float("nan")
        fnr  = fn / (fn + tp) if (fn + tp) else float("nan")
        acc  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else float("nan")
        return prec, rec, fpr, fnr, acc

    prec_tr, rec_tr, fpr_tr, fnr_tr, acc_tr = rates(tp_tr, fp_tr, fn_tr, tn_tr)
    prec_te, rec_te, fpr_te, fnr_te, acc_te = rates(tp_te, fp_te, fn_te, tn_te)

    # --- PRINTS (as requested) ---
    print("\n=== Confusion counts (positive = TRUNCATE) ===")
    print(f"TRAIN: TP={tp_tr}  FP={fp_tr}  FN={fn_tr}  TN={tn_tr}")
    print(f"TEST : TP={tp_te}  FP={fp_te}  FN={fn_te}  TN={tn_te}")

    # --- plot limits ---
    y_all = np.concatenate([y_train, yhat_train, y_test, yhat_test])
    lo, hi = float(np.min(y_all)), float(np.max(y_all))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(5.8, 5.8))

    # --- Background shading (blue/orange) ---
    # Correct quadrants (TP, TN): blue
    # Error quadrants (FP, FN): orange
    blue_bg   = "tab:blue"
    orange_bg = "tab:orange"

    # TP: x<c, y<c  (bottom-left)  -> correct (blue)
    ax.axvspan(lo, c_line, ymin=0.0, ymax=(c_line - lo) / (hi - lo), color=blue_bg, alpha=0.08)

    # TN: x>=c, y>=c (top-right)  -> correct (blue)
    ax.axvspan(c_line, hi, ymin=(c_line - lo) / (hi - lo), ymax=1.0, color=blue_bg, alpha=0.08)

    # FP: x>=c, y<c (bottom-right) -> error (orange)
    ax.axvspan(c_line, hi, ymin=0.0, ymax=(c_line - lo) / (hi - lo), color=orange_bg, alpha=0.08)

    # FN: x<c, y>=c (top-left) -> error (orange)
    ax.axvspan(lo, c_line, ymin=(c_line - lo) / (hi - lo), ymax=1.0, color=orange_bg, alpha=0.08)

    # Identity line
    ax.plot([lo, hi], [lo, hi], color="black", lw=1)

    # Truncation lines
    ax.axvline(c_line, color="black", ls="--", lw=1)
    ax.axhline(c_line, color="black", ls="--", lw=1)

    # Scatter points
    ax.scatter(y_train, yhat_train, color="tab:blue", alpha=alpha, s=s, label="Train")
    ax.scatter(y_test,  yhat_test,  color="tab:red",  alpha=alpha, s=s, label="Test")

    ax.set_xlabel(r"$t(P)$")
    ax.set_ylabel(r"$M(P)$")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    plt.tight_layout()
    plt.show()
