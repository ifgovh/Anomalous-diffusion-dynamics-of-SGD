import math
import torch
import numpy as np

def get_layerWise_norms(net):
    w = []
    g = []
    for p in net.parameters():    
        if p.requires_grad:
            w.append(p.view(-1).norm().detach().cpu().numpy())
            g.append(p.grad.view(-1).norm().detach().cpu().numpy())
    return w, g


def get_grads(model): 
    # wrt data at the current step
    res = []
    for p in model.parameters():
        if p.requires_grad:
            res.append(p.grad.view(-1))
    grad_flat = torch.cat(res)
    return grad_flat

# Corollary 2.4 in Mohammadi 2014
def alpha_estimator(m, X):
    # X is N by d matrix
    N = len(X)
    n = int(N/m) # must be an integer
    Y = torch.sum(X.view(n, m, -1), 1)
    eps = np.spacing(1)
    Y_log_norm = torch.log(Y.norm(dim=1) + eps).mean()
    X_log_norm = torch.log(X.norm(dim=1) + eps).mean()
    diff = (Y_log_norm - X_log_norm) / math.log(m)
    return 1 / diff

# Corollary 2.2 in Mohammadi 2014
def alpha_estimator2(m, k, X):
    # X is N by d matrix
    N = len(X)
    n = int(N/m) # must be an integer
    Y = torch.sum(X.view(n, m, -1), 1)
    eps = np.spacing(1)
    Y_log_norm = torch.log(Y.norm(dim=1) + eps)
    X_log_norm = torch.log(X.norm(dim=1) + eps)

    # This can be implemented more efficiently by using 
    # the np.partition function, which currently doesn't 
    # exist in pytorch: may consider passing the tensor to np
    
    Yk = torch.sort(Y_log_norm)[0][k-1]
    Xk = torch.sort(X_log_norm)[0][m*k-1]
    diff = (Yk - Xk) / math.log(m)
    return 1 / diff