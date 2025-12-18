# src/harvest_S.jl
# Stage 1: harvest S and tau from generator representatives
# Threaded per representative (robust indexing)

using PauliPropagation
using DataFrames
using CSV

using Plots
using LaTeXStrings

const HAVE_STATSPLOTS = let ok = false
    try
        @eval using StatsPlots
        ok = true
    catch
        ok = false
    end
    ok
end

# Helper: extract an Int key from whatever PauliSum iteration returns
# In this codebase, iterating a PauliSum yields integer-encoded terms (UInt16/UInt24/...)
# even if the PauliSum was constructed from PauliString objects.
@inline _pauli_key(p) = p isa PauliString ? Int(p.term) : Int(p)

function harvest_S_from_generators(
    L1_steps::Int,
    layer::Vector{Gate},
    thetas::Vector{Float64},
    initial_psum::PauliSum;
    nqubits::Int,
    c1::Float64 = 1e-4,
    max_weight::Real = Inf,
    savefile::Union{Nothing,String} = nothing,
    collect_stats::Bool = false,
    plot_times_file::Union{Nothing,String} = nothing,
)

    # ---- extract representatives as PauliString objects ----
    reps = PauliString[]
    for (p, _) in initial_psum
        term = p isa PauliString ? p.term : p
        push!(reps, PauliString(nqubits, term, 1.0))
    end

    nT = Threads.nthreads()

    # ---- bucket-local accumulators (NOT indexed by threadid) ----
    S_buckets   = [Set{Int}() for _ in 1:nT]
    tau_buckets = [Dict{Int,Bool}() for _ in 1:nT]

    times_buckets = collect_stats ? [[Float64[] for _ in 1:L1_steps] for _ in 1:nT] : nothing
    mems_buckets  = collect_stats ? [[Float64[] for _ in 1:L1_steps] for _ in 1:nT] : nothing

    # ---- worker for a single representative ----
    function harvest_one_rep!(bucket::Int, rep::PauliString)
        psum = PauliSum([rep])

        current_terms = Set{Int}()
        for (p, _) in psum
            ip = _pauli_key(p)
            push!(current_terms, ip)
            tau_buckets[bucket][ip] = false
        end

        for ℓ in 1:L1_steps
            previous_terms = copy(current_terms)

            if collect_stats
                t0 = Base.time_ns()
                bytes = @allocated propagate!(
                    layer, psum, thetas;
                    min_abs_coeff = c1,
                    max_weight = max_weight,
                )
                t1 = Base.time_ns()
                push!(times_buckets[bucket][ℓ], (t1 - t0) / 1e9)
                push!(mems_buckets[bucket][ℓ],  bytes)
            else
                propagate!(layer, psum, thetas; min_abs_coeff=c1, max_weight=max_weight)
            end

            empty!(current_terms)
            for (p, _) in psum
                ip = _pauli_key(p)
                push!(current_terms, ip)
                if !haskey(tau_buckets[bucket], ip)
                    tau_buckets[bucket][ip] = false
                end
            end

            for t in setdiff(previous_terms, current_terms)
                tau_buckets[bucket][t] = true
            end
        end

        for k in keys(tau_buckets[bucket])
            push!(S_buckets[bucket], k)
        end
    end

    # ---- threaded loop (SAFE indexing) ----
    Threads.@threads for i in eachindex(reps)
        bucket = (i - 1) % nT + 1
        harvest_one_rep!(bucket, reps[i])
    end

    # ---- merge buckets ----
    S_global = Set{Int}()
    tau_map  = Dict{Int,Bool}()

    for b in 1:nT
        union!(S_global, S_buckets[b])
        for (k, v) in tau_buckets[b]
            tau_map[k] = get(tau_map, k, false) || v
        end
    end

    S = collect(S_global)

    # ---- save Step 1 CSV ----
    if savefile !== nothing
        filename = endswith(savefile, ".csv") ? savefile : string(savefile, ".csv")
        tau_vals = [get(tau_map, t, false) ? 1 : 0 for t in S]
        CSV.write(filename, DataFrame(observable=S, tau=tau_vals))
    end

    # ---- stats + plot ----
    stats = nothing
    if collect_stats
        times = [Float64[] for _ in 1:L1_steps]
        mems  = [Float64[] for _ in 1:L1_steps]

        for ℓ in 1:L1_steps
            for b in 1:nT
                append!(times[ℓ], times_buckets[b][ℓ])
                append!(mems[ℓ],  mems_buckets[b][ℓ])
            end
        end

        if plot_times_file !== nothing && L1_steps >= 2
            xs = Int[]
            ys = Float64[]
            for ℓ in 2:L1_steps
                append!(xs, fill(ℓ, length(times[ℓ])))
                append!(ys, 1000 .* times[ℓ])
            end

            if HAVE_STATSPLOTS
                plt = StatsPlots.violin(
                    xs, ys;
                    xlabel = L"\ell",
                    ylabel = L"t\,(\mathrm{ms})",
                    legend = false,
                )
                StatsPlots.savefig(plt, plot_times_file)
            else
                plt = Plots.scatter(
                    xs, ys;
                    xlabel = L"\ell",
                    ylabel = L"t\,(\mathrm{ms})",
                    legend = false,
                )
                Plots.savefig(plt, plot_times_file)
            end
        end

        stats = (times, mems)
    end

    return S, tau_map, stats
end
