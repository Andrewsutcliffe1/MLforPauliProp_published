# src/build_dataset.jl
# High-level pipeline for constructing the training dataset (with capped Step-2 reps + robust threaded Step-2)

using PauliPropagation
using CSV
using DataFrames
using Random
using Base.Threads
using Printf

# Optional progress bar for sequential Step 2
const HAVE_PROGRESS = let ok = false
    try
        @eval using ProgressMeter
        ok = true
    catch
        ok = false
    end
    ok
end

# ---------------------------------
# Step 2 helpers: chunked CSV + merge
# ---------------------------------

"""
Compute Step-2 features by chunking observables and writing one CSV per chunk.

This is thread-safe because each task writes to its own file and does not touch shared Dicts.
Returns the vector of chunk file paths that were written.

- chunks: number of chunks (often = Threads.nthreads() or 4*Threads.nthreads() for load-balancing)
- progress_every: per-chunk print frequency (rare prints only)
"""
function compute_targets_L2_to_chunked_csv(
    L2_steps::Int,
    layer::Vector{Gate},
    observables::Vector{<:PauliString},
    thetas::Vector{Float64};
    c2::Float64,
    max_weight::Real,
    out_prefix::String,
    chunks::Int = Threads.nthreads(),
    progress_every::Int = 5000,
    verbatim::Bool = true,
)
    nobs = length(observables)
    nobs == 0 && return String[]

    # Precompute final observables once
    list_final = final_observables(observables[1].nqubits)
    nfeat = 1 + length(list_final)

    # Avoid spawning more chunks than work
    chunks = max(1, min(chunks, nobs))

    # Balanced contiguous index ranges
    edges = round.(Int, range(0, nobs, length = chunks + 1))
    ranges = [ (edges[k] + 1):(edges[k + 1]) for k in 1:chunks if edges[k] < edges[k + 1] ]

    tasks = Vector{Task}(undef, length(ranges))
    outfiles = Vector{String}(undef, length(ranges))

    for (ci, r) in enumerate(ranges)
        outfile = @sprintf("%s_%02d.csv", out_prefix, ci)
        outfiles[ci] = outfile

        tasks[ci] = Threads.@spawn begin
            m = length(r)
            ids = Vector{Int}(undef, m)
            X   = Matrix{Float64}(undef, m, nfeat)

            local_done = 0
            for (kk, i) in enumerate(r)
                obs = observables[i]
                ids[kk] = Int(obs.term)

                psum = trotter_time_evolution(
                    L2_steps, layer, obs, thetas;
                    min_abs_coeff = c2,
                    max_weight = max_weight,
                )

                @inbounds begin
                    X[kk, 1] = overlapwithzero(psum)
                    for j in 1:length(list_final)
                        X[kk, 1 + j] = overlapwithpaulisum(list_final[j], psum)
                    end
                end

                local_done += 1
                if progress_every > 0 && (local_done % progress_every == 0)
                    verbatim && println("Step 2 chunk ", ci, ": ", local_done, "/", m)
                end
            end

            df = DataFrame()
            df.observable = ids
            for j in 1:nfeat
                df[!, Symbol("f$j")] = @view X[:, j]
            end
            CSV.write(outfile, df)

            return nothing
        end
    end

    foreach(wait, tasks)

    verbatim && println("Step 2 chunk files written: ", join(outfiles, ", "))
    return outfiles
end

"""
Merge per-chunk rep CSVs into a single rep CSV (deterministic sorting by observable).
"""
function merge_rep_chunks(chunk_files::Vector{String}, save_rep::String; verbatim::Bool = true)
    isempty(chunk_files) && return nothing

    filename = endswith(save_rep, ".csv") ? save_rep : string(save_rep, ".csv")
    dfs = DataFrame[]
    for f in chunk_files
        push!(dfs, CSV.read(f, DataFrame))
    end
    df = vcat(dfs...)
    sort!(df, :observable)
    CSV.write(filename, df)
    verbatim && println("Merged rep file: ", filename)
    return filename
end

"""
Load rep CSV into Dict{Int,Vector{Float64}} for downstream code that expects this type.
"""
function load_rep_csv_to_dict(save_rep::String)
    filename = endswith(save_rep, ".csv") ? save_rep : string(save_rep, ".csv")
    df = CSV.read(filename, DataFrame)
    colnames = names(df)

    rep_results = Dict{Int,Vector{Float64}}()
    sizehint!(rep_results, nrow(df))

    for row in eachrow(df)
        obs_int = Int(row[colnames[1]])
        feats = Vector{Float64}(undef, length(colnames) - 1)
        @inbounds for j in 2:length(colnames)
            feats[j - 1] = Float64(row[colnames[j]])
        end
        rep_results[obs_int] = feats
    end
    return rep_results
end

# ---------------------------------
# Main pipeline
# ---------------------------------

"""
    build_dataset_D(nqubits, topology; kwargs...)

Adds:
- max_reps / reps_seed: cap number of Step-2 orbit representatives
- step2_parallel / step2_chunks / progress_every_step2: robust parallel Step-2 using chunk CSVs
"""
function build_dataset_D(
    nqubits::Int,
    topology::Vector{Tuple{Int,Int}};
    include_single_Z::Bool = true,
    include_double_Z::Bool = true,
    L1::Int = 32,
    L2::Int = 2,
    c1::Float64 = 1e-4,
    c2::Float64 = 0.0,
    T::Float64 = 1.0,
    J::Float64 = 2.0,
    h::Float64 = 1.0,
    max_weight::Real = Inf,
    verbatim::Bool = true,

    save_step1::Union{Nothing,String} = nothing,
    save_rep::Union{Nothing,String} = nothing,
    save_full::Union{Nothing,String} = nothing,

    skip_step1::Bool = false,
    skip_step2::Bool = false,

    # Step-1 instrumentation
    collect_stats_step1::Bool = false,
    plot_step1_times::Union{Nothing,String} = nothing,

    # Sequential progress controls
    show_progress::Bool = true,
    progress_every::Int = 200,

    # ---- NEW: cap Step-2 reps ----
    max_reps::Int = 200_000,
    reps_seed::Union{Nothing,Int} = nothing,

    # ---- NEW: threaded Step-2 ----
    step2_parallel::Bool = true,
    step2_chunks::Int = Threads.nthreads(),     # or e.g. 4*Threads.nthreads()
    progress_every_step2::Int = 5000,
)

    # --- symmetries from topology ---
    symmetries = automorphism_group(nqubits, topology)

    # --- generator set P ---
    generators = build_P(
        nqubits;
        include_single_Z = include_single_Z,
        include_double_Z = include_double_Z,
    )

    # infer packed Pauli integer type
    PT = typeof(generators[1]).parameters[1]
    verbatim && println("Using Pauli integer type: ", PT)

    # reduce P to orbit reps P̃
    P_tilde = orbit_representatives(generators, symmetries)
    P_tilde_PT = [convert_pauli_type(p, PT) for p in P_tilde]
    psum_generators = PauliSum(P_tilde_PT)

    # build one trotter layer + angles
    trotterlayer = tfitrottercircuit(nqubits, 1, topology = topology)
    dt = T / L2
    thetas = define_thetas(trotterlayer, dt; J = J, h = h)

    # ============================
    # Step 1: harvest S and tau
    # ============================
    local S::Vector{Int}
    local tau_map::Dict{Int,Bool}
    local stats1

    if skip_step1
        save_step1 === nothing && error("skip_step1=true requires save_step1")
        verbatim && println("Step 1 skipped; loading S and tau_map from ", save_step1)

        filename = endswith(save_step1, ".csv") ? save_step1 : string(save_step1, ".csv")
        isfile(filename) || error("Could not find saved step-1 file: $filename")

        df_S = CSV.read(filename, DataFrame)
        S = Vector{Int}(df_S.observable)
        tau_map = Dict{Int,Bool}()
        for (obs, tau_val) in zip(S, df_S.tau)
            tau_map[obs] = (tau_val == 1) || (tau_val == true)
        end
        stats1 = nothing
    else
        verbatim && println("Step 1: harvesting observables (L1=", L1, ", c1=", c1, ")")
        S, tau_map, stats1 = harvest_S_from_generators(
            L1,
            trotterlayer,
            thetas,
            psum_generators;
            nqubits = nqubits,
            c1 = c1,
            max_weight = max_weight,
            savefile = save_step1,
            collect_stats = collect_stats_step1,
            plot_times_file = plot_step1_times,
        )
    end

    # Reduce S to orbit reps S̃
    reps_S = orbit_representatives(S, nqubits, symmetries)

    # Cap the number of reps used in Step 2
    if length(reps_S) > max_reps
        rng = reps_seed === nothing ? Random.default_rng() : MersenneTwister(reps_seed)
        idx = randperm(rng, length(reps_S))[1:max_reps]
        reps_S = reps_S[idx]
        verbatim && println("Capping Step-2 reps: using ", length(reps_S),
                            " / ", max_reps, " (seed=", reps_seed, ")")
    else
        verbatim && println("Step-2 reps: ", length(reps_S), " (<= max_reps=", max_reps, ")")
    end

    reps_PT = [convert_pauli_type(p, PT) for p in reps_S]

    # ============================
    # step 2: compute features
    # ============================
    local rep_results::Dict{Int,Vector{Float64}}
    stats2 = nothing

    if skip_step2
        save_rep === nothing && error("skip_step2=true requires save_rep")
        verbatim && println("Step 2 skipped; loading representative features from ", save_rep)
        rep_results = load_rep_csv_to_dict(save_rep)
    else
        verbatim && println("Step 2: computing features (L2=", L2, ", c2=", c2, ") on ", length(reps_PT), " reps")

        if step2_parallel
            save_rep === nothing && error("step2_parallel=true requires save_rep (to merge chunk files)")

            # Avoid more chunks than work for tiny runs
            local_chunks = min(step2_chunks, max(1, length(reps_PT)))

            out_prefix = replace(save_rep, ".csv" => "") * "_part"
            chunk_files = compute_targets_L2_to_chunked_csv(
                L2,
                trotterlayer,
                reps_PT,
                thetas;
                c2 = c2,
                max_weight = max_weight,
                out_prefix = out_prefix,
                chunks = local_chunks,
                progress_every = progress_every_step2,
                verbatim = verbatim,
            )

            merge_rep_chunks(chunk_files, save_rep; verbatim = verbatim)

            # Downstream expects Dict
            rep_results = load_rep_csv_to_dict(save_rep)

        else
            # Sequential fallback (ProgressMeter supported)
            rep_results = Dict{Int,Vector{Float64}}()

            if !isempty(reps_PT)
                list_final = final_observables(nqubits)
                nobs = length(reps_PT)

                if show_progress && HAVE_PROGRESS
                    pb = ProgressMeter.Progress(nobs; dt=0.25, desc="Step 2")
                    for (k, obs) in enumerate(reps_PT)
                        psum = trotter_time_evolution(
                            L2, trotterlayer, obs, thetas;
                            min_abs_coeff = c2,
                            max_weight = max_weight,
                        )
                        feats = Vector{Float64}(undef, 1 + length(list_final))
                        feats[1] = overlapwithzero(psum)
                        for (j, fin) in enumerate(list_final)
                            feats[1 + j] = overlapwithpaulisum(fin, psum)
                        end
                        rep_results[Int(obs.term)] = feats
                        ProgressMeter.next!(pb)
                    end
                else
                    for (k, obs) in enumerate(reps_PT)
                        psum = trotter_time_evolution(
                            L2, trotterlayer, obs, thetas;
                            min_abs_coeff = c2,
                            max_weight = max_weight,
                        )
                        feats = Vector{Float64}(undef, 1 + length(list_final))
                        feats[1] = overlapwithzero(psum)
                        for (j, fin) in enumerate(list_final)
                            feats[1 + j] = overlapwithpaulisum(fin, psum)
                        end
                        rep_results[Int(obs.term)] = feats
                        if show_progress && (k % progress_every == 0 || k == nobs)
                            println("Step 2: ", k, "/", nobs)
                        end
                    end
                end
            end

            # Optional write
            if save_rep !== nothing && !isempty(rep_results)
                filename = endswith(save_rep, ".csv") ? save_rep : string(save_rep, ".csv")
                ids   = collect(keys(rep_results))
                feats = collect(values(rep_results))
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
        end
    end

    # ============================
    # Extend reps -> full orbits
    # ============================
    full_results = extend_results_to_orbits(
        rep_results,
        reps_PT,
        symmetries;
        savefile = save_full,
    )

    return (
        Dataset_L1 = rep_results,
        Dataset_L1_extended = full_results,
        S = S,
        tau_map = tau_map,
        stats_step1 = stats1,
        stats_step2 = stats2,
    )
end
