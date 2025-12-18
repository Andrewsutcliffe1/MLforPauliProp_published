"""eda_helpers_pandas.py

Pandas-first EDA helpers for PauliPropagation datasets.

Expected columns in your merged dataframe:
- observable : int  (base-4 packed Pauli string)
- tau        : int  (0 = untruncated, 1 = truncated)
- f1         : float (overlap with |0...0><0...0|)

Notes
-----
- Weight definitions used here:
  * pauli_weight  = #(non-identity)  = count(digit != 0)
  * xy_weight     = #(X or Y)        = count(digit in {1,2})
  where the per-qubit digit is base-4: I=0, X=1, Y=2, Z=3.
"""

from __future__ import annotations

from pathlib import Path
from math import erf, sqrt
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Folder utilities
# -------------------------------------------------------------------

def list_datasets_with_rep(base_dir: Union[str, Path]) -> list[str]:
    """List subfolders that contain a rep.csv file."""
    base_dir = Path(base_dir)
    return sorted([p.name for p in base_dir.iterdir() if p.is_dir() and (p / "rep.csv").exists()])


# -------------------------------------------------------------------
# Loading / merging (optional convenience)
# -------------------------------------------------------------------

def read_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read a CSV with sensible defaults for these datasets."""
    path = Path(path)
    defaults = dict(low_memory=False)
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)


def merge_step1_and_rep(step1_csv: Union[str, Path], rep_csv: Union[str, Path]) -> pd.DataFrame:
    """    Merge step-1 'S.csv' (observable,tau) with 'rep.csv' (observable,f1,f2,...).

    Returns a single DataFrame.
    """
    S = read_csv(step1_csv)
    rep = read_csv(rep_csv)
    return S.merge(rep, on="observable", how="inner", validate="one_to_one")


# -------------------------------------------------------------------
# Stats (pandas-first)
# -------------------------------------------------------------------

def tau_counts(df: pd.DataFrame, tau_col: str = "tau") -> pd.Series:
    """Return counts of tau values."""
    return df[tau_col].value_counts(dropna=False).sort_index()


def print_tau_counts(df: pd.DataFrame, tau_col: str = "tau") -> None:
    """Print the number of tau=0 and tau=1 entries."""
    counts = tau_counts(df, tau_col=tau_col)
    c0 = int(counts.get(0, 0))
    c1 = int(counts.get(1, 0))
    print(f"Number of entries with tau = 0: {c0}")
    print(f"Number of entries with tau = 1: {c1}")
    extra = counts.drop(labels=[0, 1], errors="ignore")
    if len(extra) > 0:
        print("Other tau values present:")
        print(extra.to_string())


def count_zero_overlap(df: pd.DataFrame, overlap_col: str = "f1") -> int:
    """Return the number of rows where overlap is exactly 0."""
    return int((df[overlap_col] == 0).sum())


def print_zero_overlap(df: pd.DataFrame, overlap_col: str = "f1") -> None:
    """Print the number of rows where overlap is exactly 0."""
    print(f"Number of entries with {overlap_col} = 0: {count_zero_overlap(df, overlap_col=overlap_col)}")


def second_smallest_abs_overlap(
    df: pd.DataFrame,
    overlap_col: str = "f1",
    *,
    exclude_zero: bool = True,
) -> Optional[float]:
    """    Return the second smallest |overlap| value.

    By default, zeros are excluded.
    Returns None if fewer than 2 eligible values exist.
    """
    s = df[overlap_col].astype(float).abs()
    if exclude_zero:
        s = s[s != 0.0]
    two = s.nsmallest(2)
    if len(two) < 2:
        return None
    return float(two.iloc[1])


def print_second_smallest_abs_overlap(
    df: pd.DataFrame,
    overlap_col: str = "f1",
    *,
    exclude_zero: bool = True,
) -> None:
    """Print the second smallest |overlap| value."""
    v = second_smallest_abs_overlap(df, overlap_col=overlap_col, exclude_zero=exclude_zero)
    if v is None:
        print("Not enough eligible values to determine the second smallest |overlap|.")
    else:
        ztxt = "excluding 0" if exclude_zero else "including 0"
        print(f"Second smallest |{overlap_col}| ({ztxt}): {v}")


# -------------------------------------------------------------------
# Pauli decoding / weights (vectorized)
# -------------------------------------------------------------------

def infer_n_qubits_from_observable_ids(obs_ids: Union[pd.Series, np.ndarray, Iterable[int]]) -> int:
    """Infer the smallest n such that max_id < 4^n."""
    arr = np.asarray(list(obs_ids), dtype=np.int64)
    if arr.size == 0:
        return 0
    max_id = int(arr.max())
    if max_id <= 0:
        return 1
    n = int(np.floor(np.log(max_id) / np.log(4))) + 1
    while 4 ** n <= max_id:
        n += 1
    return n


def _digits_base4_matrix(obs: np.ndarray, n_qubits: int) -> np.ndarray:
    """Return an (N, n_qubits) matrix of base-4 digits for each observable."""
    obs = obs.astype(np.int64, copy=False)
    powers = (4 ** np.arange(n_qubits, dtype=np.int64))
    return (obs[:, None] // powers[None, :]) % 4


def pauli_weight(df: pd.DataFrame, *, n_qubits: Optional[int] = None, observable_col: str = "observable") -> pd.Series:
    """Compute weight = #(non-identity) as a pandas Series (vectorized)."""
    obs = df[observable_col].astype(np.int64).to_numpy()
    if n_qubits is None:
        n_qubits = infer_n_qubits_from_observable_ids(obs)
    digits = _digits_base4_matrix(obs, n_qubits)
    w = (digits != 0).sum(axis=1)
    return pd.Series(w, index=df.index, name="pauli_weight")


def xy_weight(df: pd.DataFrame, *, n_qubits: Optional[int] = None, observable_col: str = "observable") -> pd.Series:
    """Compute XY-weight = #(X or Y) as a pandas Series (vectorized)."""
    obs = df[observable_col].astype(np.int64).to_numpy()
    if n_qubits is None:
        n_qubits = infer_n_qubits_from_observable_ids(obs)
    digits = _digits_base4_matrix(obs, n_qubits)
    w = ((digits == 1) | (digits == 2)).sum(axis=1)
    return pd.Series(w, index=df.index, name="xy_weight")


def with_weights(
    df: pd.DataFrame,
    *,
    n_qubits: Optional[int] = None,
    observable_col: str = "observable",
    add_pauli_weight: bool = True,
    add_xy_weight: bool = True,
) -> pd.DataFrame:
    """Return a copy of df with (optionally) pauli_weight and xy_weight columns added."""
    out = df.copy()
    if add_pauli_weight:
        out["pauli_weight"] = pauli_weight(out, n_qubits=n_qubits, observable_col=observable_col)
    if add_xy_weight:
        out["xy_weight"] = xy_weight(out, n_qubits=n_qubits, observable_col=observable_col)
    return out


# -------------------------------------------------------------------
# Empirical CDF
# -------------------------------------------------------------------

def empirical_cdf(x: Union[pd.Series, np.ndarray, Iterable[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical CDF F(x) = P(X <= x). Returns sorted x and F values."""
    arr = np.asarray(list(x), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])
    x_sorted = np.sort(arr)
    n = x_sorted.size
    F = np.arange(1, n + 1, dtype=float) / n
    return x_sorted, F


# -------------------------------------------------------------------
# Plot style
# -------------------------------------------------------------------

mpl.rcParams.update({
    "figure.figsize": (4.0, 2.0),
    "figure.dpi": 180,
    "savefig.dpi": 300,

    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",

    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,

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

    "axes.grid": True,

    "lines.linewidth": 1.5,
    "lines.markersize": 4,

    "legend.frameon": False,

    "axes.unicode_minus": False,
    "text.usetex": False,
})


# -------------------------------------------------------------------
# CDF plots
# -------------------------------------------------------------------

def plot_cdf_weight(
    df: pd.DataFrame,
    *,
    weight: str = "pauli_weight",
    n_qubits: Optional[int] = None,
    observable_col: str = "observable",
    color: str = "C0",
) -> None:
    """
    Plot the empirical CDF of a Pauli weight computed from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing Pauli observables.
    weight : {"pauli_weight", "xy_weight"}
        Which weight to plot.
    n_qubits : int, optional
        Number of qubits (required if weight must be computed).
    observable_col : str
        Column containing Pauli integer encodings.
    color : str
        Matplotlib color.
    """
    work = df

    # --- compute weight if missing ---
    if weight not in work.columns:
        if weight == "pauli_weight":
            work = work.assign(
                pauli_weight=pauli_weight(
                    work,
                    n_qubits=n_qubits,
                    observable_col=observable_col,
                )
            )
        elif weight == "xy_weight":
            work = work.assign(
                xy_weight=xy_weight(
                    work,
                    n_qubits=n_qubits,
                    observable_col=observable_col,
                )
            )
        else:
            raise KeyError(
                f"Column '{weight}' not found and not a known computable weight."
            )

    values = work[weight].to_numpy()

    # --- empirical CDF ---
    x = np.sort(values)
    F = np.arange(1, len(x) + 1) / len(x)

    # --- plot ---
    plt.figure()
    plt.step(x, F, where="post", color=color)
    if weight == "xy_weight":
        plt.xlabel(r"$W_{xy}(P)$")
    else :
        plt.xlabel(r"$W(P)$")
    plt.ylabel(r"$\Pr(W(P) \leq x)$")
    plt.tight_layout()
    plt.show()


def plot_cdf_overlap(
    df: pd.DataFrame,
    *,
    overlap_col: str = "f1",
    color: str = "C0",
    fit_gaussian: bool = True,
) -> None:
    """
    CDF of the raw overlap (not absolute, not log), from a DataFrame.
    Optionally fits a Gaussian and overlays its CDF.
    """
    values = df[overlap_col].to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    # --- empirical CDF ---
    x = np.sort(values)
    F = np.arange(1, len(x) + 1) / len(x)

    plt.figure()
    plt.step(x, F, where="post", color=color)

    # --- Gaussian fit & CDF (default) ---
    if fit_gaussian:
        mu = np.mean(values)
        sigma = np.std(values, ddof=1)

        if sigma > 0:
            xg = np.linspace(x.min(), x.max(), 512)
            z = (xg - mu) / (sigma * sqrt(2))
            Fg = 0.5 * (1.0 + np.vectorize(erf)(z))

            plt.plot(
                xg,
                Fg,
                color="black",
                linestyle=":",
                linewidth=1.2,
                label=fr"$\mu={mu:.2e},\,\sigma={sigma:.2e}$",
            )

    # --- labels ---
    plt.xlabel(
        r"$\mathrm{Tr}\!\left[(V^\dagger)^{L_2} P V^{L_2} |0^n\rangle\langle 0^n|\right]$"
    )
    plt.ylabel(r"$\Pr(X \leq x)$")
    plt.legend(
        loc="upper left",
        fontsize=6,
        frameon=False,
        handlelength=2.0,
    )


    plt.tight_layout()
    plt.show()



def plot_cdf_abs_overlap(
    df: pd.DataFrame,
    *,
    overlap_col: str = "f1",
    epsilon: float = 1e-15,
    color: str = "C0",
) -> None:
    """
    CDF of |overlap| with controlled epsilon and log-scale, from a DataFrame.
    """
    arr = np.abs(df[overlap_col].to_numpy(dtype=float))
    arr = np.maximum(arr, float(epsilon))

    x = np.sort(arr)
    F = np.arange(1, len(x) + 1) / len(x)

    plt.figure()
    plt.step(x, F, where="post", color=color, linewidth=1.6)
    plt.xscale("log")
    plt.xlabel(
        r"$\left|\mathrm{Tr}\!\left[(V^\dagger)^{L_2} P V^{L_2} |0^n\rangle\langle 0^n|\right]\right|$"
    )
    plt.ylabel(r"$\Pr(X \leq x)$")
    plt.tight_layout()
    plt.show()



# -------------------------------------------------------------------
# Overlap vs weight plot
# -------------------------------------------------------------------


def plot_overlap_vs_weight(
    df: pd.DataFrame,
    *,
    overlap_col: str = "f1",
    weight: str = "pauli_weight",
    n_qubits: Optional[int] = None,
    observable_col: str = "observable",
    color: str = "C0",
    alpha: float = 0.6,
    s: float = 12.0,
) -> None:
    """
    Scatter plot of overlap as a function of Pauli weight.
    """
    work = df

    # --- compute weight if missing ---
    if weight not in work.columns:
        if weight == "pauli_weight":
            work = work.assign(
                pauli_weight=pauli_weight(
                    work,
                    n_qubits=n_qubits,
                    observable_col=observable_col,
                )
            )
        elif weight == "xy_weight":
            work = work.assign(
                xy_weight=xy_weight(
                    work,
                    n_qubits=n_qubits,
                    observable_col=observable_col,
                )
            )
        else:
            raise KeyError(
                f"Column '{weight}' not found and not a known computable weight."
            )

    x = work[weight].to_numpy()
    y = work[overlap_col].to_numpy()

    plt.figure()
    plt.scatter(
        x,
        y,
        s=s,
        alpha=alpha,
        color=color,
        edgecolors="none",
    )

    # --- labels ---
    if weight == "xy_weight":
        plt.xlabel(r"$W_{xy}(P)$")
    else:
        plt.xlabel(r"$W(P)$")

    plt.ylabel(
        r"$\mathrm{Tr}\!\left[(V^\dagger)^{L_2} P V^{L_2} |0^n\rangle\langle 0^n|\right]$"
    )

    plt.tight_layout()
    plt.show()

