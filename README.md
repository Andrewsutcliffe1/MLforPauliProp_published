# The Demon model: 
## A Graph Neural Network Surrogate for Pauli Propagation

This repository contains the code and experiments accompanying the report  
**“Graph Neural Network Surrogate for PauliPropagation.jl”**.

We implement a Graph Neural Network (GNN) surrogate model called **Demon**.  
It is able to predict future expectation values of Pauli strings under
Trotterized Heisenberg evolution.  
It is designed to improve truncation strategies in Sparse Pauli Dynamics (SPD),
reducing systematic bias while maintaining computational tractability.

## Project Overview

Classical simulation of many-body quantum dynamics is a central challenge in
near-term quantum computing. Sparse Pauli Dynamics (SPD) mitigates the
exponential growth of operators by truncating Pauli strings during evolution,
but this introduces bias.

In this project, we propose a model that predicts the future contribution of a
Pauli string during propagation.

These predictions can be used as an informed truncation criterion for Pauli terms during evolution.

## Repository Structure

- `PauliPropagation.jl` # Julia code for Pauli propagation experiments  
- `Trainings/` # Training on different configurations and with different architectures  
- `datasets/` # Multiple datasets generated and tested
- `implemented_improvements/` # Improvement on the generation pipeline
- `models/` # Saved trained model parameters  
- `GNN.ipynb` # Main notebook for training and evaluating the Demon model  
- `StateVecEvol.py` # State vector evolution utilities  
- `comparison.ipynb` # Comparison with standard truncation methods  
- `demon_model.py` # Demon model implementation  

### PauliPropagation.jl 
**Data Generation**
- The `simpleDatasetProduction.ipynb` notebook shows how data and labels are generated.
- Pauli strings are collected during truncated SPD evolution (L1) using `PauliPropagation.jl`.
- A representative subset of the collected Pauli strings is propagated exactly
  (or with a very small truncation) for a few additional Trotter layers (L2).
- The regression target is the logarithm of the overlap between the evolved Pauli
  operator and the computational ground state.

### Training
- Details on training, validation, and testing are explained both in the report
  and in `GNN.ipynb`.
- This section contains plots and numerical values of the loss function over
  10 epochs of training for different configurations:
  - the Demon model trained on the 100,004 Pauli dataset `training_100004`
    and the 99,699 Pauli dataset `training_99699`;
  - the Demon model without linear lifting from the ablation study
    `training_ablation(noLifting)`;
  - the neural network baseline model trained on the 100,004 Pauli dataset
    `training_NN`.
- Moreover, hyperparameter tuning grids with corresponding validation loss
  evaluations are reported for both the Demon model and the neural network:
  `NN_grid_search_results.csv`, `grid_search_results.csv`.

### datasets
- We generated different datasets based on their size, on the depth of the
  first Pauli propagation (L1), and on the accuracy of the corresponding labels.
- All datasets produced during the project are reported here.

### implemented_improvements
- This section documents improvements considered towards the end of the project.
- There are three notebook navigating through the different stages of the project (Dataset, EDA, and and Training[unfinished])

### models
- The Demon model is a Graph Neural Network using a message-passing step to
  aggregate information across node attributes before applying standard MLP
  layers.
- The architecture is invariant under automorphisms of the interaction graph.
- All successfully trained model checkpoints are saved here.

## Results

The flagship model (**Demon**) demonstrates very high predictive accuracy on
unseen Pauli strings:

- **Test MSE:** ~2.9 × 10⁻⁴  
- **R² score:** 0.99998  

Ablation studies show that:
- GNNs significantly outperform non-graph baselines;
- linear lifting and normalization play a critical stabilizing role;
- including truncated Pauli strings in the dataset leads to improved performance.

---

### Usage

### Running experiments
- The `GNN.ipynb` notebook shows the full workflow to train and evaluate the
  surrogate model.
- `demon_model.py` can be used to evaluate the performance of the model on new,
  unseen datasets.
- `comparison.ipynb` contains the first analysis comparing the informed
  truncation method based on **Demon** with the standard truncation approach.
  Further analyses remain to be developed.
- `StateVecEvol.py` contains code provided by a researcher in our lab to evaluate
  model performance against a ground-truth state-vector simulation.

### Dependencies
- Python
- Julia (for `PauliPropagation.jl`)
  
#### Julia Dependencies
Julia packages and versions are specified in `Project.toml` and `Manifest.toml`.

To install all Julia dependencies:
```bash
cd PauliPropagation.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Install python packages in requirements.txt

## Authors
- Andrew Sutcliffe  
- Matteo Casiraghi  
- Hugo Izadi  

## References
See the project report for full references.
