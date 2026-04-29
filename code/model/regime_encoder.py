import torch
import torch.nn as nn

class RegimeEncoder(nn.Module):
    """
    Stronger regime encoder using temporal convolution
    """
    def __init__(self, n_vertex, embed_dim=16):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * n_vertex, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):

        x = x.mean(dim=1, keepdim=True)  
        x = self.temporal_conv(x)     

        x = x.mean(dim=2)               
        x = x.flatten(start_dim=1)       

        regime = self.fc(x)
        return regime
