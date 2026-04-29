import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import torch

# ================= GSO =================
def calc_gso(dir_adj, gso_type):
    n_vertex = dir_adj.shape[0]

    if not sp.issparse(dir_adj):
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id_mat = sp.identity(n_vertex, format='csc')

    # symmetric adjacency
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
    adj = adj + id_mat
  
    row_sum = adj.sum(axis=1).A1
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D = sp.diags(d_inv_sqrt)

    gso = D.dot(adj).dot(D)

    return gso

# ================= Chebyshev =================
def calc_chebynet_gso(gso):
    if not sp.issparse(gso):
        gso = sp.csc_matrix(gso)

    id_mat = sp.identity(gso.shape[0], format='csc')
    eigval_max = norm(gso, 2)

    if eigval_max >= 2:
        gso = gso - id_mat
    else:
        gso = 2 * gso / eigval_max - id_mat

    return gso

# ================= FINAL METRICS =================
def evaluate_metric(model, data_iter, scaler):

    model.eval()
    mae_list, rmse_list, mape_list = [], [], []

    with torch.no_grad():
        for x, y in data_iter:
            output = model(x)

            if isinstance(output, tuple):
                pred = output[0]
            else:
                pred = output

            pred = scaler.inverse_transform(
                pred.cpu().numpy().reshape(-1, pred.shape[-1])
            )
            true = scaler.inverse_transform(
                y.cpu().numpy().reshape(-1, y.shape[-1])
            )

            diff = np.abs(true - pred)
            mae_list.append(np.mean(diff))
            rmse_list.append(np.sqrt(np.mean(diff**2)))

            # masked MAPE (paper)
            mask = true > 10
            if np.sum(mask) == 0:
                continue

            mape_list.append(
                np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
            )

    return np.mean(mae_list), np.mean(rmse_list), np.mean(mape_list)
