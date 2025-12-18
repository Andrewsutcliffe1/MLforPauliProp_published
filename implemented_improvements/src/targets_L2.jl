# Step 2: feature computation for the dataset

using PauliPropagation
using DataFrames
using CSV

"""
    final_observables(numQbits)

Return the list of Pauli observables used for feature extraction.  By
default this includes all single‑Z operators `Z_i` and all double‑Z
operators `Z_i Z_j` for `1 ≤ i < j ≤ numQbits`.  These observables are
used to compute overlaps in the Level‑2 propagation stage.

Note: this function relies on `single_Z_paulis` and `double_Z_paulis`
being defined in the current scope (see `generators_P.jl`).  When
using these helpers as part of a larger program be sure to include
that file or otherwise make those functions available.
"""
function final_observables(numQbits::Int)
    singles = single_Z_paulis(numQbits)
    doubles = double_Z_paulis(numQbits)
    return vcat(singles, doubles)
end

"""
    compute_targets_L2(L2_steps, layer, observables, thetas;
                       c2=0.0, max_weight=Inf, savefile=nothing)

For each observable in `observables`, perform `L2_steps` Trotter
evolution layers and compute a feature vector consisting of the
overlap with the computational basis projector |0…0⟩⟨0…0| followed by
overlaps with each Pauli in `final_observables(n_qubits)`.  The
truncation threshold for this stage is `c2`, which defaults to 0 to
ensure no terms are discarded.  Returns a dictionary mapping the
integer encoding of each input observable to its feature vector.

When `savefile` is supplied the results are written to CSV with
columns `observable, f1, f2, …`.
"""
function compute_targets_L2(
    L2_steps::Int,
    layer::Vector{Gate},
    observables::Vector{<:PauliString},
    thetas::Vector{Float64};
    c2::Float64 = 0.0,
    max_weight::Real = Inf,
    savefile::Union{Nothing,String} = nothing,
)
    first_obs = observables[1]
    n_qubits  = first_obs.nqubits
    list_final = final_observables(n_qubits)
    results = Dict{Int,Vector{Float64}}()
    for obs in observables
        # Propagate exactly L2_steps layers; by default c2=0 ensures no truncation
        psum = trotter_time_evolution(
            L2_steps,
            layer,
            obs,
            thetas;
            min_abs_coeff = c2,
            max_weight = max_weight,
        )
        # Build feature vector: overlap with |0…0⟩ followed by overlaps with each final observable
        features = Float64[]
        push!(features, overlapwithzero(psum))
        for final_observable in list_final
            push!(features, overlapwithpaulisum(final_observable, psum))
        end
        results[Int(obs.term)] = features
    end
    # Optionally write to disk
    if savefile !== nothing && !isempty(results)
        filename = endswith(savefile, ".csv") ? savefile : string(savefile, ".csv")
        ids   = collect(keys(results))
        feats = collect(values(results))
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
    return results
end