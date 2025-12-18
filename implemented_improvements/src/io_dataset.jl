# Dataset I/O helpers

using DataFrames
using CSV
using PauliPropagation

"""
    extend_results_to_orbits(rep_results, representatives, symmetries; savefile=nothing)

Extend a dictionary of representative results to all members of their
symmetry orbits.  The input `rep_results` should map the integer
encoding of a representative Pauli string to its feature vector.  The
vector `representatives` must contain the same set of representatives
(typically as `PauliString`s) and `symmetries` is the automorphism
group used to generate orbits.  Returns a new dictionary mapping the
integer encodings of all orbit members to the corresponding feature
vector.  If `savefile` is provided a CSV is written with columns
`observable` and `f1, f2, …`.
"""
function extend_results_to_orbits(
    rep_results::Dict{Int,Vector{Float64}},
    representatives::Vector{<:PauliString},
    symmetries::Vector{Vector{Int}};
    savefile::Union{Nothing,String} = nothing,
)
    # Extended results mapping: integer encoding -> feature vector.
    full_results = Dict{Int,Vector{Float64}}()
    if isempty(representatives)
        return full_results
    end
    # Determine the number of qubits
    n_qubits = representatives[1].nqubits
    # Construct the list of final observables used in the feature vectors.
    # We reconstruct this list here rather than relying on external state,
    # to ensure that permutations of the feature vector are correct when
    # extending to all orbit members.  The ordering must match the one used
    # in compute_targets_L2: overlap with |0…0⟩ followed by overlaps with
    # single‑Z operators Z_i (i=1..n) then double‑Z operators Z_i Z_j (i<j).
    singles = [PauliString(n_qubits, :Z, i) for i in 1:n_qubits]
    doubles = PauliString[]
    for i in 1:n_qubits-1
        for j in i+1:n_qubits
            push!(doubles, PauliString(n_qubits, [:Z, :Z], [i, j]))
        end
    end
    list_final = vcat(singles, doubles)
    # Build a lookup: observable term -> index in list_final
    obs_index = Dict{Int,Int}()
    for (idx, obs) in enumerate(list_final)
        obs_index[Int(obs.term)] = idx
    end
    # Precompute, for each symmetry, a mapping from feature indices to their
    # permuted positions.  Each mapping is a vector `idx_map` of length
    # length(list_final) such that idx_map[k] gives the new position of the
    # feature originally at position k (1‑based) in the final observable list.
    num_feats = length(list_final)
    index_maps = Vector{Vector{Int}}(undef, length(symmetries))
    for (sidx, sym) in enumerate(symmetries)
        idx_map = zeros(Int, num_feats)
        for old_idx in 1:num_feats
            # Permute the base‑4 vector representation of the original observable
            original = list_final[old_idx]
            perm_vec = permute_pauli_vector(pauli_to_int_vector(n_qubits, original), sym)
            permuted_obs = int_vector_to_pauli(perm_vec)
            new_idx = obs_index[Int(permuted_obs.term)]
            idx_map[old_idx] = new_idx
        end
        index_maps[sidx] = idx_map
    end
    # Populate full_results by permuting the representative features for each
    # symmetry‑related observable.
    for rep in representatives
        rep_term = Int(rep.term)
        if !haskey(rep_results, rep_term)
            @warn "No entry in rep_results for representative with term = $rep_term"
            continue
        end
        base_feats = rep_results[rep_term]
        # assign features to representative itself without permutation
        full_results[rep_term] = base_feats
        # For each symmetry, generate the orbit member and permute features
        for (sidx, sym) in enumerate(symmetries)
            # apply the symmetry to the representative PauliString
            perm_vec = permute_pauli_vector(pauli_to_int_vector(n_qubits, rep), sym)
            obs = int_vector_to_pauli(perm_vec)
            obs_term = Int(obs.term)
            # skip the representative itself to avoid overwriting
            if obs_term == rep_term
                continue
            end
            # Build a new feature vector: first entry (overlap with |0…0⟩)
            # remains unchanged, subsequent entries are permuted according to
            # index_map.
            new_feats = Vector{Float64}(undef, length(base_feats))
            new_feats[1] = base_feats[1]
            idx_map = index_maps[sidx]
            for old_idx in 1:num_feats
                new_pos = idx_map[old_idx] + 1 # offset by 1 for the |0…0⟩ feature
                new_feats[new_pos] = base_feats[old_idx + 1]
            end
            full_results[obs_term] = new_feats
        end
    end
    # Optionally write out the extended dataset
    if savefile !== nothing && !isempty(full_results)
        filename = endswith(savefile, ".csv") ? savefile : string(savefile, ".csv")
        ids   = collect(keys(full_results))
        feats = collect(values(full_results))
        perm  = sortperm(ids)
        ids   = ids[perm]
        feats = feats[perm]
        nfeat = length(feats[1])
        df = DataFrame()
        df.observable = ids
        for j in 1:nfeat
            df[!, Symbol("f$j")] = [v[j] for v in feats]
        end
        CSV.write(filename, df)
    end
    return full_results
end

"""
    save_S(S, tau_map, filename)

Write the results of Step 1 to a CSV file.  `S` is a vector of
observable encodings and `tau_map` maps each encoding to a Boolean
indicator of truncation.  The output file has columns `observable`
and `tau` where `tau` is 0 for untruncated terms and 1 for terms
truncated during Level‑1 evolution.
"""
function save_S(S::Vector{Int}, tau_map::Dict{Int,Bool}, filename::String)
    tau_vals = [tau_map[t] ? 1 : 0 for t in S]
    df = DataFrame(observable = S, tau = tau_vals)
    filename_csv = endswith(filename, ".csv") ? filename : string(filename, ".csv")
    CSV.write(filename_csv, df)
    return nothing
end