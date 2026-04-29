import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicAdjacency(nn.Module):
    def __init__(self, num_nodes, seq_len, hidden_dim=32, num_heads=4):
        super(DynamicAdjacency, self).__init__()
        self.num_heads = num_heads

        # Multi-head temporal projections
        self.temporal_proj = nn.ModuleList([
            nn.Linear(seq_len, hidden_dim)
            for _ in range(num_heads)
        ])

        # Multi-head node projections
        self.node_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        x = x.squeeze(1)         
        x = x.permute(0, 2, 1)   
        adj_heads = []
        for i in range(self.num_heads):
            # Temporal encoding per head
            h = torch.relu(self.temporal_proj[i](x))
            h = torch.relu(self.node_proj[i](h))
            # Similarity per head
            A = torch.matmul(h, h.transpose(1, 2))
            # Normalize
            A = F.softmax(A, dim=-1)
            adj_heads.append(A)
        A_dynamic = torch.stack(adj_heads, dim=0).mean(dim=0)

        return A_dynamic
