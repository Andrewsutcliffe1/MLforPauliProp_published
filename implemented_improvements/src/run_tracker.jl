# Helpers for tracking dataset generation runs

using JSON3
using Dates

"""
    random_run_id(len=5)

Generate a random alphanumeric string of length `len`.  This helper is
used to create unique directory names for runs or to tag output files.
"""
function random_run_id(len::Int = 5)
    chars = ['a':'z'; 'A':'Z'; '0':'9']
    return String([rand(chars) for _ in 1:len])
end

"""
    sanitize_number(x)

Convert a number into a file‑system friendly string by replacing
decimal points with the character `p`.  For example, `0.1` becomes
`"0p1"`.  Non‑numeric inputs are converted to strings unchanged.
"""
function sanitize_number(x)
    return replace(string(x), "." => "p")
end

"""
    json_safe_number(x)

Return a number that can be serialised by JSON3.  If `x` is finite it
is returned directly; otherwise a string representation such as
`"Inf"`, `"-Inf"` or `"NaN"` is returned.  Non‑numeric inputs are
passed through unchanged.
"""
function json_safe_number(x::Real)
    xf = float(x)
    return isfinite(xf) ? x : string(x)
end
json_safe_number(x) = x

"""
    load_existing_runs(tracker_file)

Load an existing JSON tracker file.  If the file does not exist,
cannot be parsed or is empty, an empty vector is returned.  The
tracker file stores an array of objects, each describing a previous
dataset generation run.
"""
function load_existing_runs(tracker_file::String)
    if !isfile(tracker_file)
        return Any[]
    end
    if filesize(tracker_file) == 0
        return Any[]
    end
    try
        open(tracker_file, "r") do io
            return JSON3.read(io, Vector{Any})
        end
    catch
        @warn "Could not parse existing tracker file $tracker_file, starting a new one."
        return Any[]
    end
end

"""
    save_topology_run(tracker_file, runinfo; append=true)

Append a run description `runinfo` to the JSON tracker file
`tracker_file`.  When `append` is true (the default) the existing
entries are preserved; otherwise the file is overwritten with a new
array containing only `runinfo`.
"""
function save_topology_run(tracker_file::String, runinfo::Dict{String,Any}; append::Bool = true)
    existing = append ? load_existing_runs(tracker_file) : Any[]
    push!(existing, runinfo)
    open(tracker_file, "w") do io
        JSON3.write(io, existing)
    end
    return nothing
end