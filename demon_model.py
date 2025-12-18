import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool


# GRAPH HELPERS FOR BUILDING INPUT DATA

# Convert integer → base-4 Pauli index sequence
def pauli_digits_from_int(x: int, n: int):
    """
    Convert integer → list of base-4 digits [0,1,2,3] of length n.
    These digits correspond to Pauli operators: I,X,Y,Z.
    """
    digits = []
    for _ in range(n):
        digits.append(x % 4)
        x //= 4
    return torch.tensor(digits, dtype=torch.long)

# Pauli feature embedding 
def pauli_features(P_indices):
    """
    Convert P_indices (0,1,2,3) to 3-bit Pauli vectors.
    """
    pauli_map = torch.tensor([
        [0,0,0],  # I
        [1,0,0],  # X
        [0,1,0],  # Y
        [0,0,1],  # Z
    ], dtype=torch.float)

    return pauli_map[P_indices]

# 1D LINE GRAPH 
def make_ising_line_graph(P_indices):
    n = P_indices.size(0)
    x = pauli_features(P_indices)
    src = torch.arange(n - 1)
    dst = src + 1
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src])
    ])
    return Data(x=x, edge_index=edge_index)

# 1D RING GRAPH
def make_ising_ring_graph(P_indices):
    n = P_indices.size(0)
    x = pauli_features(P_indices)
    src = torch.arange(n)
    dst = (src + 1) % n
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src])
    ])
    return Data(x=x, edge_index=edge_index)

# 2D SQUARE LATTICE GRAPH
def make_ising_2d_graph(P_indices, height, width):
    x = pauli_features(P_indices)
    edges_src = []
    edges_dst = []
    for r in range(height):
        for c in range(width):
            idx = r * width + c
            # Right neighbor
            if c + 1 < width:
                nbr = idx + 1
                edges_src += [idx, nbr]
                edges_dst += [nbr, idx]
            # Down neighbor
            if r + 1 < height:
                nbr = (r + 1) * width + c
                edges_src += [idx, nbr]
                edges_dst += [nbr, idx]
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


# dataset builder with topology selection

def build_graph_dataset(pauli_ints,
                        num_qubits,
                        topology="line",
                        height=None,
                        width=None):
    """
    Build a list of PyG graph objects for inference.

    Args:
        pauli_ints : list/array of integers representing Pauli strings
        num_qubits : length of Pauli string (number of nodes)
        topology   : one of {"line", "ring", "2d"}
        height,width : required for 2D topology

    Returns:
        graphs : list[Data] each containing x and edge_index
    """
    graphs = []

    for x in pauli_ints:

        # Convert integer → Pauli index vector
        P_indices = pauli_digits_from_int(x, num_qubits).view(-1)

        # Select topology
        if topology == "line":
            data = make_ising_line_graph(P_indices)

        elif topology == "ring":
            data = make_ising_ring_graph(P_indices)

        elif topology == "2d":
            assert height is not None and width is not None, \
                "height and width must be provided for 2D topology"
            assert height * width == num_qubits, \
                "height * width must equal num_qubits"
            data = make_ising_2d_graph(P_indices, height, width)

        else:
            raise ValueError(f"Unknown topology: '{topology}'")

        # NO LABELS ATTACHED
        graphs.append(data)

    return graphs



# MODEL ARCHITECTURE: MESSAGE PASSING + FFN BLOCKS

class TopologyPropagation(MessagePassing):
    """
    Message passing layer:
    Implements: x = x + sum_j MLP(J * LayerNorm(x_j))
    """
    def __init__(self, dim):
        super().__init__(aggr="add")
        self.norm = nn.LayerNorm(dim)

        self.msg_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, edge_index, J):
        self.J = J

        # PreNorm: LN(x)
        x_norm = self.norm(x)

        # Nonlinear message passing
        msg = self.propagate(edge_index, x=x_norm)

        # Residual update
        return x + msg

    def message(self, x_j):
        # Apply nonlinear per-edge transformation
        return self.msg_mlp(self.J * x_j)


class NodeMLP(nn.Module):
    def __init__(self, dim, hidden_dim=48, dropout=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm(x)
        return x + self.mlp(x_norm)


class DemonLayer(nn.Module):
    def __init__(self, dim, hidden_dim=48, dropout=0.1):
        super().__init__()
        self.conv = TopologyPropagation(dim)
        self.mlp = NodeMLP(dim, hidden_dim, dropout)

    def forward(self, x, edge_index, J):
        x = self.conv(x, edge_index, J)
        x = self.mlp(x)
        return x


class Demon(nn.Module):
    """
    Main DEMON GNN model
    """
    def __init__(self,
                 num_layers,
                 base_dim=3,
                 hidden_dim=96,
                 expansion_ratio=8,
                 dropout=0.0):
        super().__init__()

        model_dim = num_layers * expansion_ratio
        self.input_proj = nn.Linear(base_dim, model_dim)

        self.layers = nn.ModuleList([
            DemonLayer(model_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.readout_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1)
        )

    def forward(self, x, edge_index, J, batch=None):
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x, edge_index, J)

        if batch is None:
            # Original single-graph behavior
            graph_rep = x.mean(dim=0)
            return self.readout_mlp(graph_rep).squeeze()
        else:
            # Batched behavior: compute mean for each graph separately
            graph_rep = global_mean_pool(x, batch)  # [batch_size, model_dim]
            return self.readout_mlp(graph_rep).squeeze(-1)  # [batch_size]

def create_batched_data(x_list, edge_index):
    """
    Create a batched PyTorch Geometric Data object from multiple node feature matrices.
    
    Args:
        x_list: List of node feature tensors, each [num_nodes, base_dim]
        edge_index: Edge index tensor [2, num_edges] (same topology for all graphs)
        num_nodes_per_graph: Number of nodes in each graph (assumes all graphs have same size)
    
    Returns:
        x_batched: Concatenated node features [batch_size * num_nodes, base_dim]
        edge_index_batched: Batched edge indices [2, batch_size * num_edges]
        batch: Batch assignment vector [batch_size * num_nodes]
    """
    

    data_list = [Data(x=x, edge_index=edge_index) for x in x_list]
    
    batched_data = Batch.from_data_list(data_list)
    
    return batched_data.x, batched_data.edge_index, batched_data.batch

# USER-FRIENDLY INFERENCE WRAPPER

class DemonWrapper:
    """
    Wrapper for easy inference
    """
    def __init__(self,
                 checkpoint_path=None,
                 num_layers=2,
                 base_dim=3,
                 hidden_dim=96,
                 expansion_ratio=8,
                 dropout=0.0,
                 device="cpu"):

        self.device = device

        self.model = Demon(
            num_layers=num_layers,
            base_dim=base_dim,
            hidden_dim=hidden_dim,
            expansion_ratio=expansion_ratio,
            dropout=dropout
        ).to(device)

        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            self.model.eval()

    @torch.no_grad()
    def forward(self, x, edge_index, J=1, batch=None):
        """
        Forward with optional batching.
        
        Returns:
            Single scalar if batch=None, otherwise tensor of shape [batch_size] on CPU
        """
        result = self.model(
            x.to(self.device),
            edge_index.to(self.device),
            J,
            batch.to(self.device) if batch is not None else None
        )
        
        if batch is None:
            return result.cpu().item()  # Single scalar
        else:
            return result.cpu()


def load_demon(checkpoint_path, **kwargs):
    return DemonWrapper(checkpoint_path=checkpoint_path, **kwargs)



## TUTORIAL

# from demon_model import (
# build_graph_dataset,
# load_demon
# )

# graphs = build_graph_dataset(
#     [0],
#     20,
#     topology="ring"
# )
# print(graphs)
# checkpoint_path = "models/best_demon_len=100004_qbits=20_{Gen=true_L1=32_cp=0.00123418_w=Inf}_{L2=2_cp=1.0e-15}.pth"


# demon = load_demon(
#     checkpoint_path,
#     num_layers=2,
#     device="cpu"
# )

# model = demon.model

# n_params = sum(p.numel() for p in model.parameters())
# print(n_params)

# param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
# buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
# total_mb = (param_bytes + buffer_bytes) / 1024**2

# print(f"{n_params/1e6:.2f}M parameters, {total_mb:.1f} MB")


# pred = demon.forward(graphs[0].x, graphs[0].edge_index, J=1)
# print(pred)
