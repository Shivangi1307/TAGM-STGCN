import torch
import torch.nn as nn


class GraphMemory(nn.Module):
    """
    Temporal Adaptive Graph Memory (TAGM)
    Maintains exponential moving average of dynamic adjacency
    """
    def __init__(self, num_nodes, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.num_nodes = num_nodes
        self.register_buffer("memory", None)

    def forward(self, A_dyn):
        A_mean = A_dyn.mean(dim=0)  # average over batch

        if self.memory is None:
            self.memory = A_mean.detach()
        # EMA update
        self.memory = (
            self.momentum * self.memory +
            (1 - self.momentum) * A_mean.detach()
        )

        A_mem = self.memory.unsqueeze(0).expand(A_dyn.size(0), -1, -1)
        return A_mem
