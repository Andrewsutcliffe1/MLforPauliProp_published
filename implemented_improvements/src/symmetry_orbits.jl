# Symmetry and orbit helpers

using Graphs

# Import basic encoding functions.  When these helpers are used in
# concert with other files one should ensure that `pauli_to_int_vector`
# and `int_vector_to_pauli` are available in the module scope.  This
# file does not explicitly `include` them to keep coupling loose.

"""
    automorphism_group(n_qubits, topology)

Return the automorphism group of the qubit connectivity graph defined
by `topology`.  The result is a vector of permutations, where each
permutation is itself a vector such that `perm[i]` gives the new
location of qubit `i` under that symmetry.  The connectivity graph
treated here is undirected.
"""
function automorphism_group(n_qubits::Int, topology::Vector{Tuple{Int,Int}})
    g = SimpleGraph(n_qubits)
    for (u, v) in topology
        Graphs.add_edge!(g, u, v)
    end

    automorphisms = Graphs.Experimental.all_isomorph(
        g, g, Graphs.Experimental.VF2()
    )

    group = Vector{Vector{Int}}()
    for iso in automorphisms
        perm = zeros(Int, n_qubits)
        for (src, dst) in iso
            perm[src] = dst
        end
        push!(group, perm)
    end
    return group
end

"""
    permute_pauli_vector(v, permutation)

Apply a permutation of qubit indices to a base‑4 encoded Pauli vector
`v`.  The returned vector has the same length as `v` and entries
corresponding to the Pauli operators on permuted qubits.
"""
function permute_pauli_vector(v::Vector{Int}, permutation::Vector{Int})
    n = length(v)
    new_v = zeros(Int, n)
    for i in 1:n
        target_index = permutation[i]
        new_v[target_index] = v[i]
    end
    return new_v
end

"""
    canonical_representative(obs, n, symmetries)

Find the lexicographically smallest Pauli string in the orbit of
`obs` under the provided symmetry group.  This helper accepts both
integer encodings and `PauliString` objects.  The output is always a
`PauliString` whose coefficient is set to unity.
"""
function canonical_representative(obs::Union{Vector{PauliSum},Vector{PauliString},Vector{PauliString{UInt32,Float64}},Integer}, n::Int, symmetries::Vector{Vector{Int}})
    # Convert to integer encoding via the base‑4 vector representation
    current_vec = pauli_to_int_vector(n, obs)
    min_vec     = current_vec
    for sym in symmetries
        candidate = permute_pauli_vector(current_vec, sym)
        if candidate < min_vec
            min_vec = candidate
        end
    end
    return int_vector_to_pauli(min_vec)
end

"""
    orbit(P_tilde, symmetries)

Generate the full orbit of a representative Pauli string `P_tilde` under
the symmetry group.  The returned vector contains `PauliString`
instances corresponding to every symmetry‑related observable.
"""
function orbit(P_tilde::PauliString, symmetries::Vector{Vector{Int}})
    n        = P_tilde.nqubits
    base_vec = pauli_to_int_vector(n, P_tilde)
    orbit_set = Set{Vector{Int}}()
    for sym in symmetries
        candidate = permute_pauli_vector(base_vec, sym)
        push!(orbit_set, candidate)
    end
    return [int_vector_to_pauli(v) for v in orbit_set]
end

"""
    orbit_size(P_tilde, symmetries)

Return the size of the orbit of `P_tilde` under the symmetry group.
This is computed by enumerating the orbit; callers that already have
the orbit can avoid this overhead by taking the length directly.
"""
orbit_size(P_tilde::PauliString, symmetries::Vector{Vector{Int}}) = length(orbit(P_tilde, symmetries))

"""
    orbit_representatives(candidates, symmetries)

Return unique canonical representatives for a list of Pauli observables.
The input may be provided as a vector of `PauliString`s or a vector
of integer encodings.  Each representative is chosen to be the
lexicographically smallest element in its orbit.
"""
function orbit_representatives(candidates::Vector{PauliString}, symmetries::Vector{Vector{Int}})
    n = candidates[1].nqubits
    unique_reps_vec = Set{Vector{Int}}()
    for obs in candidates
        rep_pauli = canonical_representative(Int(obs.term), n, symmetries)
        rep_vec   = pauli_to_int_vector(n, rep_pauli)
        push!(unique_reps_vec, rep_vec)
    end
    return [int_vector_to_pauli(v) for v in unique_reps_vec]
end

function orbit_representatives(candidates::Vector{Int}, n::Int, symmetries::Vector{Vector{Int}})
    unique_reps_vec = Set{Vector{Int}}()
    for obs_int in candidates
        rep_pauli = canonical_representative(obs_int, n, symmetries)
        rep_vec   = pauli_to_int_vector(n, rep_pauli)
        push!(unique_reps_vec, rep_vec)
    end
    return [int_vector_to_pauli(v) for v in unique_reps_vec]
end