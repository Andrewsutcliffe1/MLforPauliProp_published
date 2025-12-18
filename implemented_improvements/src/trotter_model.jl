# Low level Trotterisation utilities

using PauliPropagation

"""
    define_thetas(circuit, dt; J=2.0, h=1.0)

Compute the rotation angles for a single Trotter layer.  The angles
are determined by the strengths of the RZZ and RX interactions with
coupling constants `J` and `h` respectively, and the time step `dt`.
The output is a vector of length equal to the number of tunable
parameters in `circuit`, compatible with `propagate!`.
"""
function define_thetas(
    circuit::Vector{Gate},
    dt::Float64;
    J::Float64 = 2.0,
    h::Float64 = 1.0,
)
    rzz_indices = getparameterindices(circuit, PauliRotation, [:Z, :Z])
    rx_indices  = getparameterindices(circuit, PauliRotation, [:X])
    nparams     = countparameters(circuit)
    thetas      = zeros(Float64, nparams)
    thetas[rzz_indices] .= -J * dt * 2
    thetas[rx_indices]  .=  h * dt * 2
    return thetas
end

"""
    trotter_time_evolution(steps, layer, observable, thetas; min_abs_coeff, max_weight)

Apply `steps` layers of Trotterised time evolution to an initial
observable.  The observable may be a `PauliString` or a `PauliSum`.
Internally this routine repeatedly calls `propagate!` from the
`PauliPropagation` package and collects the resulting `PauliSum`.
Arguments `min_abs_coeff` and `max_weight` are passed through to
`propagate!` to control truncation and weight cut‑offs.
"""
function trotter_time_evolution(
    steps::Int,
    layer::Vector{Gate},
    observable::Union{PauliSum, PauliString},
    thetas::Vector{Float64};
    min_abs_coeff::Float64 = 1e-6,
    max_weight::Real       = 10,
    collect_stats::Bool    = false,
)
    psum = observable isa PauliString ? PauliSum([observable]) : deepcopy(observable)
    if collect_stats
        times = zeros(Float64, steps)
        mems  = zeros(Float64, steps)
        for s in 1:steps
            t0 = Base.time_ns()
            bytes = @allocated propagate!(layer, psum, thetas; min_abs_coeff = min_abs_coeff, max_weight = max_weight)
            t1 = Base.time_ns()
            times[s] = (t1 - t0) / 1e9
            mems[s]  = bytes
        end
        return psum, times, mems
    else
        for _ in 1:steps
            propagate!(layer, psum, thetas; min_abs_coeff = min_abs_coeff, max_weight = max_weight)
        end
        return psum
    end
end