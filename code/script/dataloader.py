import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

def load_adj(dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()

    if dataset_name == 'metr-la':
        n_vertex = 207
    elif dataset_name == 'pems-bay':
        n_vertex = 325
    elif dataset_name == 'pemsd7-m':
        n_vertex = 228
    else:
        raise ValueError("Unknown dataset")

    return adj, n_vertex

def data_transform(data, n_his, n_pred, device):
    """
    Transform raw traffic matrix → STGCN input format
    x: [B, 1, T, N]
    y: [B, 12, N]
    """
  
    n_route = data.shape[1]
    n_samples = data.shape[0] - n_his - n_pred + 1
  
    x = np.zeros((n_samples, 1, n_his, n_route))
    y = np.zeros((n_samples, n_pred, n_route))

    for i in range(n_samples):
        x[i, :, :, :] = data[i:i+n_his].reshape(1, n_his, n_route)
        y[i, :, :] = data[i+n_his:i+n_his+n_pred]

    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    return x, y
