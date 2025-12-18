"""
EDA helpers for PauliPropagation datasets (pandas-first).
"""

from .eda_helpers import (
    # I/O
    list_datasets_with_rep,
    read_csv,
    merge_step1_and_rep,

    # stats
    tau_counts,
    print_tau_counts,
    count_zero_overlap,
    print_zero_overlap,
    second_smallest_abs_overlap,
    print_second_smallest_abs_overlap,

    # weights
    infer_n_qubits_from_observable_ids,
    pauli_weight,
    xy_weight,
    with_weights,

    # CDF helpers
    empirical_cdf,
    plot_cdf_weight,
    plot_cdf_overlap,
    plot_cdf_abs_overlap,

    # plots
    plot_overlap_vs_weight,
)

__all__ = [
    "list_datasets_with_rep",
    "read_csv",
    "merge_step1_and_rep",
    "tau_counts",
    "print_tau_counts",
    "count_zero_overlap",
    "print_zero_overlap",
    "second_smallest_abs_overlap",
    "print_second_smallest_abs_overlap",
    "infer_n_qubits_from_observable_ids",
    "pauli_weight",
    "xy_weight",
    "with_weights",
    "empirical_cdf",
    "plot_cdf_weight",
    "plot_cdf_overlap",
    "plot_cdf_abs_overlap",
    "plot_overlap_vs_weight",
]
