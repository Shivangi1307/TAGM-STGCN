import torch
import torch.nn as nn
from code.model import layers
from .dynamic_adj import DynamicAdjacency
from .graph_memory import GraphMemory
from .regime_encoder import RegimeEncoder

class TAGM_STGCN(nn.Module):
def **init**(self, args, blocks, n_vertex):
super().**init**()

```
    # ===== REGIME ENCODER =====
    self.regime_encoder = RegimeEncoder(n_vertex, embed_dim=16)

    self.regime_to_alpha = nn.Linear(16, 1)
    self.regime_to_beta  = nn.Linear(16, 1)

    self.args = args
    self.n_vertex = n_vertex

    self.st_blocks = nn.ModuleList()
    for l in range(len(blocks)-3):
        self.st_blocks.append(
            layers.STConvBlock(
                args.Kt,
                args.Ks,
                n_vertex,
                blocks[l][-1],
                blocks[l+1],
                args.act_func,
                args.enable_bias,
                args.droprate
            )
        )

    self.A_static = args.gso

    self.dynamic_adjs = nn.ModuleList([
        DynamicAdjacency(n_vertex, args.n_his,
                         num_heads=args.num_heads)
        for _ in range(len(self.st_blocks))
    ])

    self.memories = nn.ModuleList([
        GraphMemory(n_vertex, momentum=0.9)
        for _ in range(len(self.st_blocks))
    ])

    Ko = args.n_his - (len(blocks)-3)*2*(args.Kt-1)

    self.output = layers.OutputBlock(
        Ko,
        blocks[-3][-1],
        blocks[-2],
        blocks[-1][0],
        n_vertex,
        args.act_func,
        args.enable_bias,
        args.droprate
    )

def forward(self, x):

    B = x.size(0)
    dyn_graphs = []
    mem_graphs = []

    x_input = x

    for i, block in enumerate(self.st_blocks):

        A_static = self.A_static.unsqueeze(0).expand(B, -1, -1)

        if self.args.use_dynamic == 1:
            A_dyn = self.dynamic_adjs[i](x_input)
            dyn_graphs.append(A_dyn)

            A_mem = self.memories[i](A_dyn)
            mem_graphs.append(A_mem)

            A_final = 0.7*A_static + 0.2*A_dyn + 0.1*A_mem
        else:
            A_final = A_static

        x = block(x, A_final)

    x = self.output(x)

    return x, (dyn_graphs, mem_graphs)
```
