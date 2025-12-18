# Truncation Pauli Propagation Dataset

This repository contains a modular Julia implementation of the dataset generation pipeline described in the paper *Truncation Pauli Propagation*.  The goal of the pipeline is to build a dataset of Pauli strings and associated features produced by simulating Trotterized time evolution of spin systems under hardware connectivity constraints.

## Project layout

The code has been factored into small files under `src/` to mirror the logical pieces of the pipeline:

| File | Purpose |
| --- | --- |
| `src/pauli_encoding.jl` | Utilities for converting between Pauli strings and their integer/base‑4 vector encodings and for changing integer types. |
| `src/symmetry_orbits.jl` | Construction of the automorphism group of a qubit connectivity graph, canonical representative selection, orbit enumeration and size calculation. |
| `src/trotter_model.jl` | Definition of the Trotter layer rotation angles and a helper for repeated Trotter evolution of Pauli sums. |
| `src/generators_P.jl` | Construction of the generator set \(\mathcal{P}\) of single‑ and/or double‑\(Z\) Pauli strings and its reduction to orbit representatives \(\tilde{\mathcal{P}}\). |
| `src/harvest_S.jl` | Stage 1 of the pipeline: propagate \(\tilde{\mathcal{P}}\) for \(L_1\) Trotter steps while applying a truncation threshold and record every Pauli string encountered along with a truncation indicator \(\tau(P)\). |
| `src/targets_L2.jl` | Stage 2 of the pipeline: propagate each representative of \(\mathcal{S}\) exactly \(L_2\) steps without truncation and compute overlaps with the computational basis state \(|0\dots 0\rangle\) and all single‑ and double‑\(Z\) observables. |
| `src/io_dataset.jl` | Helper functions for extending representative results to full symmetry orbits and writing the results to CSV. |
| `src/run_tracker.jl` | Basic utilities for run identification and JSON log management. |
| `src/build_dataset.jl` | A high‑level wrapper that orchestrates both stages of the pipeline and exposes a single `build_dataset_D` function. |

## Installing dependencies

The code is written for Julia.  To get started, install Julia and then add the required packages:

```julia
using Pkg
Pkg.add("PauliPropagation")
Pkg.add("Graphs")
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("JSON3")
```

The PauliPropagation package implements the core Pauli sum arithmetic and Trotter propagation used throughout the pipeline.

## Running the pipeline

To generate a dataset, open a Julia REPL (or script) in this project folder, include all source files and call `build_dataset_D`:

```julia
include("src/pauli_encoding.jl")
include("src/symmetry_orbits.jl")
include("src/trotter_model.jl")
include("src/generators_P.jl")
include("src/harvest_S.jl")
include("src/targets_L2.jl")
include("src/io_dataset.jl")
include("src/run_tracker.jl")
include("src/build_dataset.jl")

# Example: 3‑qubit linear chain connectivity
nqubits  = 3
topology = [(1,2), (2,3)]

result = build_dataset_D(
    nqubits,
    topology;
    include_single_Z = true,
    include_double_Z = true,
    L1 = 4,
    L2 = 2,
    c1 = 1e-4,
    c2 = 0.0,
    T  = 1.0,
    J  = 2.0,
    h  = 1.0,
    save_step1 = "stage1.csv",
    save_rep   = "stage2_reps.csv",
    save_full  = "stage2_full.csv",
)
```

The `build_dataset_D` function returns a named tuple with four fields:

- **`Dataset_L1`** – a dictionary mapping each representative of \(\tilde{\mathcal{S}}\) to its feature vector.
- **`Dataset_L1_extended`** – the same feature vectors extended to every symmetry‑equivalent Pauli string in \(\mathcal{S}\).
- **`S`** – the set of all Pauli strings encountered during Stage 1.
- **`tau_map`** – a dictionary mapping each element of `S` to a truncation indicator (1 if the string was truncated, 0 otherwise).

If the `save_*` keywords are provided, the stage results are written to CSV files.  The Stage 1 CSV has columns `observable` (the integer encoding of the Pauli string) and `tau`.  The Stage 2 CSV files have columns `observable` followed by one column per feature (`f1`, `f2`, …).

## Customising the topology and generators

The `topology` argument is a vector of pairs `(u,v)` specifying which qubits are connected by ZZ rotations.  The functions in `symmetry_orbits.jl` automatically compute the automorphism group of this graph to reduce redundant calculations.  You can choose to include only single‑\(Z\) generators, only double‑\(Z\) generators, or both, via the `include_single_Z` and `include_double_Z` flags.

The integers `L1` and `L2` control the number of Trotter steps in Stages 1 and 2.  The parameters `c1` and `c2` set the truncation thresholds (with `c2=0` indicating no truncation), and `T`, `J`, `h` set the overall evolution time and Hamiltonian coupling strengths.

## Notes

- All Pauli strings are stored as integer encodings using up to 32 bits for \(n ≤ 16\) qubits and up to 64 bits for \(n ≤ 32\) qubits.
- The pipeline makes heavy use of the VF2 algorithm via the Graphs.jl package to identify graph symmetries.
- For large qubit counts the number of generated Pauli strings grows quickly; adjust `L1`, `c1` and `L2` accordingly to keep the dataset manageable.

For more details on the theoretical framework and dataset definitions, refer to the accompanying paper stored in this repository.