# Pauli encoding utilities

"""
    pauli_to_int_vector(n_qubits, obs_int)

Convert an integer encoding of a Pauli string into a base‑4 vector of
length `n_qubits`.  Each element of the returned vector is an integer
in `0:3` corresponding to I=0, X=1, Y=2, Z=3.  This helper is used
throughout the dataset generation pipeline to work in a canonical
integer representation.

This method is specialised for integer inputs; see the method below
for handling `PauliString` arguments directly.
"""
function pauli_to_int_vector(n_qubits::Int, obs_int::Integer)
    v = zeros(Int, n_qubits)
    x = obs_int
    for i in 1:n_qubits
        v[i] = x % 4
        x ÷= 4
    end
    return v
end

"""
    pauli_to_int_vector(n_qubits, obs::PauliString)

Convert a `PauliString` to a base‑4 vector by unpacking its internal
integer representation.  This is a thin wrapper around the integer
method defined above.
"""
function pauli_to_int_vector(n_qubits::Int, obs::PauliString)
    v = zeros(Int, n_qubits)
    x = obs.term
    for i in 1:n_qubits
        v[i] = x % 4
        x ÷= 4
    end
    return v
end

"""
    int_vector_to_pauli(v)

Reconstruct a `PauliString` from a base‑4 vector `v`.  The length of
`v` is the number of qubits and each entry should be an integer in
`0:3` denoting the corresponding Pauli operator.  The returned
`PauliString` is created with an integer type large enough to hold
the encoded basis element.
"""
function int_vector_to_pauli(v::Vector{Int})
    n = length(v)
    # Choose an integer type capable of representing all bits.  For up
    # to 16 qubits use UInt32, up to 32 qubits use UInt64 and above
    # fall back to UInt128.  This mirrors the original code.
    T = n <= 16 ? UInt32 : (n <= 32 ? UInt64 : UInt128)
    x = T(0)
    power_of_4 = T(1)
    for val in v
        x += T(val) * power_of_4
        power_of_4 *= 4
    end
    return PauliString(n, x, 1.0)
end

"""
    convert_pauli_type(ps, ::Type{NewT})

Return a copy of the `PauliString` `ps` whose internal integer
representation is stored as `NewT`.  This is useful when one needs
the small (e.g. `UInt32`) encoding for efficient storage or when
working with larger qubit counts that exceed `UInt32`.
"""
function convert_pauli_type(ps::PauliString{T,C}, ::Type{NewT}) where {T,C,NewT}
    return PauliString(ps.nqubits, NewT(ps.term), ps.coeff)
end