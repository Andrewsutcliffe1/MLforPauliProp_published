using PyCall
using PauliPropagation
# First, activate the virtual environment
const demon_py_file = let
    helpers_dir = @__DIR__
    
    py"""
    import sys
    import os
    from pathlib import Path

    helpers_dir = Path($helpers_dir)
    project_root = helpers_dir.parent
    
    # IMPORTANT: Change working directory to project root
    # This makes relative paths in demon_model work correctly
    os.chdir(str(project_root))
    
    venv_path = project_root / 'questenv'
    
    if sys.platform == 'win32':
        site_packages = venv_path / 'Lib' / 'site-packages'
    else:
        python_version = f'python{sys.version_info.major}.{sys.version_info.minor}'
        site_packages = venv_path / 'lib' / python_version / 'site-packages'
    
    sys.path.insert(0, str(site_packages))
    sys.path.insert(0, str(project_root))
    sys.prefix = str(venv_path)
    
    import demon_model
    """
    py"demon_model"
end

# mapping 0→:I, 1→:X, 2→:Y, 3→:Z
const PAULI_MAP = (:I, :X, :Y, :Z)

function pauliString_from_int(x::UInt, n::Int, coeff=1.0)
    paulis = Vector{Symbol}(undef, n)

    for i in 1:n
        digit = x % 4          # lowest base-4 digit
        paulis[i] = PAULI_MAP[digit+1]
        x ÷= 4
    end

    qinds = collect(1:n)
    return PauliString(n, paulis, qinds, coeff)
end



function demon_model_load(checkpoint_path,modelParams)
    demon = demon_py_file.load_demon(
    checkpoint_path,
    num_layers=modelParams["L2"], 
    hidden_dim=modelParams["hidden_dim"],
    expansion_ratio=modelParams["expansion_ratio"],
    device="cpu"
    )
    return demon
end

demon_model_load(
    "models/best_demon_len=8299_qbits=20_{Gen=true_L1=32_cp=0.005_w=Inf}_{L2=3_cp=1.0e-19}.pth",Dict([("qbits",20), ("L2",3), ("hidden_dim",48),("expansion_ratio",10)]))


function demonLabels(numLayers, observables, topology, dt, modelParams, checkpoint_path, scythe, min_abs_coeff)
    """
    Produces the overlap with zero after "observables" is propagated numLayers layers, using the model as truncation rule, following description in the report introduction.
    This truncates once per layer, with perfect c=0 evolution inbetween layers. It maintains an intermediateResult PauliSum at each layer, propagating it and choosing which Paulis to remove.
    This function also times the different sections, as bottlenecks are currently occuring in our pipeline.
    returns: overlap with zero of remaining paulis after numLayers propagation
    """
    numQbits = modelParams["qbits"]
    setup_time = @elapsed begin
        demon = demon_model_load(checkpoint_path, modelParams)
        circuit = tfitrottercircuit(numQbits, 1; topology=topology)
        parameters = ones(countparameters(circuit)) * dt
        intermediateResult = observables
        
        # Build graph once and reuse
        graphs = demon_py_file.build_graph_dataset(
            [1],
            numQbits,
            topology = "ring"
        )
        edge_index = graphs[1].edge_index
        
        # Pre-allocate arrays to avoid repeated allocations
        pauli_ints = Vector{UInt64}()
        to_remove = PauliSum(numQbits)  # Reusable for removals
    end
    
    total_feature_time = 0.0
    total_inference_time = 0.0
    total_removal_time = 0.0
    total_propagate_time = 0.0

    for i = 1:numLayers
        
        key_extraction_time = @elapsed begin
            # Get current keys - resize pre-allocated array
            resize!(pauli_ints, length(intermediateResult.terms))
            pauli_ints .= keys(intermediateResult.terms)
        end
        
        feature_time = @elapsed begin
            # Batch process
            pints = Vector{Any}(undef, length(pauli_ints))
            xs = Vector{Any}(undef, length(pauli_ints))
            
            # Batch convert paulis to features
            for (idx, pauli_int) in enumerate(pauli_ints)
                pints[idx] = demon_py_file.pauli_digits_from_int(pauli_int, numQbits).view(-1)
                xs[idx] = demon_py_file.pauli_features(pints[idx])
            end
        end
        total_feature_time += feature_time
        
        # Batch inference
        inference_and_removal_time = @elapsed begin
            empty!(to_remove.terms)
            kept_count = 0
            
            # Create batched data
            batch_time = @elapsed begin
                x_batched, edge_index_batched, batch_vector = demon_py_file.create_batched_data(
                    xs, edge_index
                )
            end
            
            # Single batched forward pass
            batch_inference_time = @elapsed begin
                demon_preds_torch = demon.forward(x_batched, edge_index_batched, 1, batch_vector)
                demon_preds = demon_preds_torch.numpy()
            end
            total_inference_time += batch_inference_time
            
            # Process predictions
            for (idx, pauli_int) in enumerate(pauli_ints)

                demon_pred = demon_preds[idx]
                coeff = intermediateResult.terms[pauli_int]
                if exp(demon_pred)*abs(coeff) < scythe # cutoff M(P)*c < threshold
                    
                    to_remove.terms[pauli_int] = -coeff
                else
                    kept_count += 1
                end
            end
            
            # Apply all removals
            add!(intermediateResult, to_remove)
        end
        total_removal_time += inference_and_removal_time

        println("Layer ", i, ": feature=", feature_time, "s, kept ", kept_count, 
                " paulis out of ", length(pauli_ints))

        if kept_count == 0 #exit if no paulis kept
            println("0 Paulis found at layer ", i, ", so exited, label 0")
            return 0
        end

        propagate_time = @elapsed begin
            intermediateResult = propagate(circuit, intermediateResult, parameters;
                        max_weight=Inf,
                        min_abs_coeff=min_abs_coeff, nl=1)
        end
        total_propagate_time += propagate_time
    end
    
    println("\n=== TIMING SUMMARY (OPTIMIZED) ===")
    println("Setup: ", setup_time, "s")
    println("Feature extraction: ", total_feature_time, "s")
    println("Model inference: ", total_inference_time, "s")
    println("Removal operations: ", total_removal_time, "s")
    println("Propagate: ", total_propagate_time, "s")
    println("Total: ", setup_time + total_feature_time + total_inference_time + total_removal_time + total_propagate_time, "s")
    
    println("Ended with ", length(intermediateResult), " paulis")
    print(intermediateResult)
    return overlapwithzero(intermediateResult), total_inference_time
end


function add_1!(psum::PauliSum, c::Number)
    # add to coeffs in-place
    for (k, v) in psum.terms
        psum.terms[k] += c
    end
    return psum
end



function produceLabels(numQbits, trainingPauliSum, L2::Integer, cp_thrshld, topology, dt)
    """
    Method to produce labels for all training paulis in set S, as defined in section 2 step 2 in the report. 
    PauliSum is propagated using PauliPropagation package for L2 layers all in one go.
    labels are [overlapwith0,customLabel,overlapwithZi for i in 1,2,3,...]
    Returns: labels
    """
    outputPauliSums = OrderedDict()
    circuit = tfitrottercircuit(numQbits, L2; topology=topology)
    parameters = ones(countparameters(circuit)) * dt
    #thetas = thetas_ttfi(circuit, gx=gx, gz=gz, deltat = deltat)

    k = 0
    tot = length(keys(trainingPauliSum.terms))
    # cps = [Float64[] for _ in 1:length(keys(trainingPauliSum.terms))] used to monitor the cps themselves
    overlapPaulis = []
    for idx in range(1,numQbits)
        push!(overlapPaulis,PauliString(numQbits, :Z, idx))
    end

    for i in keys(trainingPauliSum.terms)
        k += 1
        pauli = pauliString_from_int(UInt64(i), numQbits)
        #psum = propagate(circuit, pauli, thetas; max_weight=Inf, min_abs_coeff=cp_thrshld)
        psum = propagate(circuit, pauli, parameters;
                    max_weight=Inf,
                    min_abs_coeff=cp_thrshld, nl = L2)

        # push!(cps[k], abs.(values(psum.terms))...)
        cp_sum = log(sum(abs.(values(psum.terms))))
        labels = []
        push!(labels,overlapwithzero(psum))
        push!(labels,cp_sum)

        for ovrlpPauli in overlapPaulis
            push!(labels,overlapwithpaulisum(ovrlpPauli, psum))
        end
        outputPauliSums[pauli.term] = labels

        # log progress
        if k % 100 == 0
            print("\rprocessed $k/$tot")
            flush(stdout)
        end
    end
    return outputPauliSums
end


#### The following methods are used for TTFI propagation, and are under development

function thetas_ttfi(circuit; deltat=0.25, J=1, gz=0.9045, gx=1.4)
    thetas = []
    # print("finding indices")
    indices_gate_zz = getparameterindices(circuit, PauliRotation, [:Z, :Z])
    indices_gate_x = getparameterindices(circuit, PauliRotation, [:X])
    indices_gate_z = getparameterindices(circuit, PauliRotation, [:Z])
    for (idx, gate) in enumerate(circuit)
        if idx in indices_gate_zz
            coupling_strength = deltat * J
        elseif idx in indices_gate_x
            coupling_strength = deltat * gx
        elseif idx in indices_gate_z
            coupling_strength = deltat * gz
        else
            throw(ArgumentError("Gate at index $idx is neither ZZ, X nor Z rotation"))
        end
        push!(thetas, coupling_strength * 2) # e^i t P / 2 * 
        # if idx == length(circuit)//4
        #     print("1/4 way")
        # end
    end
            
    return thetas
end


# Define time evolution function
"""
Perform Pauli propagation for TTFI. Same as function from notebook, but simplified
"""
function ttfi_propagate(
    circuit,
    pauliString,
    thetas;
    max_weight=Inf,
    min_abs_coeff=0.,
    nl=1
)
    p_propagated = pauliString
    for i in 1:nl
        # Pauli propagation for a single Trotter step
        p_propagated = propagate(
            circuit, p_propagated, thetas; max_weight=max_weight, 
            min_abs_coeff=min_abs_coeff )
        # Compute the local expectation values for p_propagated
    end
    return p_propagated
end

function tiltedtfi_local_ham_pauli(
    nq::Int,
    gx::Float64,
    gz::Float64,
    topology=nothing,
    site=nothing, 
)
    if isnothing(site)
        if nq % 2 == 0
            site = Int(nq / 2)
        else
            site = Int((nq + 1) / 2)
        end
    else
        site = site
    end
    
    # check if the site is in the topology
    if topology != nothing
        max_site = maximum(Iterators.flatten(topology))
        if site ∉ Iterators.flatten(topology)
          throw(ArgumentError("Site $site is not in the topology."))
        end

        # check that number of qubits 
        if nq != max_site
          throw(ArgumentError(
              "The number of qubits $nq is not equal to the \
              maximum index $max_site in the topology."
          ))
        end
    else
        topology = [(ii, ii + 1) for ii in 1:nq-1]
    end

    # Build local energy operator
    local_energy = PauliSum(nq, PauliString(nq, :X, site, gx))
    add!(local_energy, :Z, site, gz)

    terms_site = Base.filter(x -> site in x, topology)
    for pair in terms_site
        # coefficient is split as 1 / number of terms
        add!(local_energy, [:Z, :Z], pair, 0.5)
    end

    return local_energy
end




















##################################3


function produceLabels2(numQbits, trainingPauliSum, L2::Integer, topology, dt)
    # preallocate dictionary with estimated size
    outputPauliSums = OrderedDict{typeof(pauliString_from_int(0, numQbits)), Float64}()

    circuit = tfitrottercircuit(numQbits, L2; topology=topology)
    parameters = ones(countparameters(circuit)) * dt

    k = 0
    tot = length(keys(trainingPauliSum.terms))

    for i in keys(trainingPauliSum.terms)
        k += 1
        pauli = pauliString_from_int(Int64(i), numQbits)
        psum = propagate(circuit, pauli, parameters; max_weight=Inf, min_abs_coeff=0)

        # avoid creating temporary array

        cp_sum = 0.0
        for v in values(psum.terms)
            cp_sum += abs(v)
        end
        outputPauliSums[pauli] = log(cp_sum)

        # overwrite current progress
        print("\rprocessed $(k)/$tot")
        flush(stdout)
    end

    println()  # move to new line after finishing
    return outputPauliSums
end



# function produceLabels_parallel(numQbits, trainingPauliSum, L2::Integer, topology, dt)
#     print("As")
#     pauli_type = typeof(pauliString_from_int(0, numQbits))
#     tot = length(keys(trainingPauliSum.terms))
#     results = Vector{Tuple{pauli_type, Float64}}(undef, tot)
#     keys_array = collect(keys(trainingPauliSum.terms))

#     circuit = tfitrottercircuit(numQbits, L2; topology=topology)
#     parameters = ones(countparameters(circuit)) * dt

#     # atomic counter for progress
#     counter = Atomic{Int}(0)
#     print("As")

#     @threads for idx in 1:tot
#         i = keys_array[idx]
#         pauli = pauliString_from_int(Int64(i), numQbits)
#         psum = propagate(circuit, pauli, parameters; max_weight=Inf, min_abs_coeff=0)

#         cp_sum = 0.0
#         for v in values(psum.terms)
#             cp_sum += abs(v)
#         end
#         results[idx] = (pauli, log(cp_sum))

#         # increment counter and print progress
#         c = atomic_add!(counter, 1)
#         if c % 10 == 0 || c == tot  # update every 10 steps
#             print("\rprocessed $c/$tot")
#             flush(stdout)
#         end
#     end

#     println()  # move to new line

#     # assemble OrderedDict
#     outputPauliSums = OrderedDict{pauli_type, Float64}()
#     for (pauli, val) in results
#         outputPauliSums[pauli] = val
#     end

#     return outputPauliSums
# end
