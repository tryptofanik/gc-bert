import numpy as np
import torch


def to_torch_sparse(x):
    coo = x.tocoo()
    row = torch.from_numpy(coo.row.astype(np.float64)).to(torch.long)
    col = torch.from_numpy(coo.col.astype(np.float64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)

    val = torch.from_numpy(coo.data.astype(np.float64)).to(torch.float64)

    return torch.sparse.DoubleTensor(edge_index, val, torch.Size(coo.shape))

