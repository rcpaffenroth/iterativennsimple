import torch

def bmatrix(M):
    rows = []
    for r in M:
        rows += [torch.hstack(r)]
    return torch.vstack(rows)