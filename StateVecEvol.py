import pyquest
from pyquest import Register
from pyquest.unitaries import H, X, Ry, Rx, Rz, Z, Y
from pyquest import Circuit
import numpy as np
from time import perf_counter


"""Exact state vector evolution using pyquest. First 3 functions mimic behaviour of the functions from PauliPropagation"""

def rectangletopology(nx, ny):
    topology = []

    for jj in range(ny):
        for ii in range(nx):

            if jj <= ny - 2:
                topology.append((jj * nx + ii, (jj+1) * nx + ii))

            if ii + 1 <= nx-1:
                topology.append(((jj) * nx + ii, (jj) * nx + ii + 1))
    return topology

def circletopology(n):
    topology = []
    for i in range(n-1):
        topology.append((i, i+1))
    topology.append((n-1, 0))

    return topology    

def create_ising_layer(nqubits, topology, theta_x=.1, theta_zz=.1, theta_z=0, tilted=False):

    """Creates a layer of Ising trotter for a given topology, and evolution parameters"""
    circuit = []
    for i in range(nqubits):
        circuit.append(Rx(i, theta_x))

    if tilted:
        for i in range(nqubits):
            circuit.append(Rx(i, theta_z))

    for i,j in topology:
        circuit.append(X(j, controls=[i]))
        circuit.append(Rz(j, theta_zz))
        circuit.append(X(j, controls=[i]))
    return circuit



def get_SV_expVal(nqubits,nlayers,topology,pauliInt):
    """
    returns expectation of PauliInt after exact evolution of |0><0|: equivalent to heisenberg evolution used in PauliProp
    
    :param pauliInt: Integer representing the starting pauli from paulipropagation
    """
    circuit = []
    pythTopo = [(a-1, b-1) for a, b in topology]
    # Creates the circuit as a list of gates
    circuit = create_ising_layer(nqubits, pythTopo)

    # Instantiate the circuit and the registers
    reg = Register(nqubits)
    circuit = Circuit(circuit)

    # Execute the circuit
    for _ in range(nlayers):
        reg.apply_circuit(circuit)

    # reg contains the final state vector
    temp = Register(copy_reg=reg)

    # Applies the operators to measure
    trackingGates = []
    gates = [X, Y, Z]     # 0→I, 1→X, 2→Y, 3→Z
    strgates = ["X","Y","Z"]
    for q in range(nqubits):       # qubits 0..n-1
        code = pauliInt % 4
        # base-4 digit
        if code==0: # identity
            pauliInt //= 4
            trackingGates.append("I")
            continue
        
        temp.apply_operator(gates[code-1](q))
        pauliInt //= 4
        trackingGates.append(strgates[code-1])
        if pauliInt == 0: #remaining are Identites
            trackingGates.append("I...")
            break

    # Final result of the expectation value by taking the overlap
    exp = (temp * reg).real
    print("applied circuit was:", trackingGates)
    return exp