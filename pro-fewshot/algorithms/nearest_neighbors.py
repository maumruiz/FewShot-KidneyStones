import torch

def distance_matrix(x, y=None, p = 2):
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = torch.pow(x - y, p).sum(2)
    
    return dist

def nearest_neighbors(X, k=3, p=2):
    eps = 0.000001
    dist = (distance_matrix(X, p=p) + eps) ** (1/2)
    knn = dist.topk(k, largest=False)
    return knn.values, knn.indices