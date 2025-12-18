# Construction of generator sets P and P_tilde

using PauliPropagation

"""
    single_Z_paulis(numQbits)

Return an array of single‑qubit Z operators `Z_i` on `numQbits`
qubits, represented as `PauliString`s.  The operators are ordered
from qubit 1 to `numQbits`.
"""
function single_Z_paulis(numQbits::Int)
    list_single_Z_paulis = PauliString[]
    for idx in 1:numQbits
        push!(list_single_Z_paulis, PauliString(numQbits, :Z, idx))
    end
    return list_single_Z_paulis
end

"""
    double_Z_paulis(numQbits)

Return an array of two‑qubit Z operators `Z_i Z_j` on `numQbits`
qubits, represented as `PauliString`s.  Only pairs `i<j` are
generated.
"""
function double_Z_paulis(numQbits::Int)
    list_double_Z_paulis = PauliString[]
    for i in 1:numQbits-1
        for j in i+1:numQbits
            push!(list_double_Z_paulis, PauliString(numQbits, [:Z, :Z], [i, j]))
        end
    end
    return list_double_Z_paulis
end

"""
    build_P(n; include_single_Z=true, include_double_Z=false)

Construct the generator set P (the paper's mathcal{P}) consisting of single and/or
double‑Z operators on `n` qubits.  When both flags are true the
returned vector contains all single‑Z and all double‑Z operators.
"""
function build_P(n::Int; include_single_Z::Bool = true, include_double_Z::Bool = false)
    gens = PauliString[]
    if include_single_Z
        append!(gens, single_Z_paulis(n))
    end
    if include_double_Z
        append!(gens, double_Z_paulis(n))
    end
    return gens
end

"""
    build_P_tilde(n, symmetries; include_single_Z=true, include_double_Z=false)

Compute a canonical set of generators by taking one representative from
each orbit of P (the paper's mathcal{P}) under the automorphism group of the
topology.  This function returns a vector of `PauliString`s that are
the unique representatives.  It relies on `orbit_representatives`
from the `symmetry_orbits` file.
"""
function build_P_tilde(n::Int, symmetries::Vector{Vector{Int}}; include_single_Z::Bool = true, include_double_Z::Bool = false)
    candidates = build_P(n; include_single_Z = include_single_Z, include_double_Z = include_double_Z)
    return orbit_representatives(candidates, symmetries)
end