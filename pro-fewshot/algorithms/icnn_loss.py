import torch
import torch.nn.functional as F

import util.globals as glob

def distance_matrix(x, y=None, p = 2):
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = torch.pow(x - y, p).sum(2)
    
    return dist

def nearest_neighbors(X, y=None, k=3, p=2):
    eps = 0.000001
    dist = (distance_matrix(X, y, p=p) + eps) ** (1/2)
    knn = dist.topk(k, largest=False)
    return knn.values, knn.indices

def score(origin, y_orig, target=None, y_targ=None, k=5, p=2, q=2, r=2):
    """Compute class prototypes from support samples.

    # Arguments
        X: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        y: torch.Tensor. The class of every sample
        k: int. the number of neigbhors (small k focus on local structures big k on global)
        p: int. must be a natural number, the higher is p, the lower penalization on lambda function
        q: int. must be a natural number, the higher is p, the lower penalization on omega function
        r: int. must be a natural number, the higher is r, the lower penalization on gamma function

    # Returns
        
    """
    target = origin if type(target) == type(None) else target
    y_targ = y_orig if type(y_targ) == type(None) else y_targ

    eps = 0.000001
    k = 3 if k >= target.shape[0] else k

    #min max scale by feature
    # a = target - target.min(axis=0).values
    # b = X.max(axis=0).values - X.min(axis=0).values
    # X = torch.divide(a , b+eps)

    distances, indices = nearest_neighbors(origin, target, k=k+1)
    distances = distances[:,1:]
    indices = indices[:,1:]

    # class by neighbor
    classes = y_targ[indices]
    yMatrix = y_orig.repeat(k,1).T
    scMatrix = (yMatrix == classes)*1 # Same class matrix [1 same class, 0 diff class]
    dcMatrix = (scMatrix)*(-1)+1 # Different class matrix [negation of scMatrix]

    ### Normalizing distances between neighbords
    dt = distances.T
    nd = (dt - dt.min(axis=0).values) / ( (dt.max(axis=0).values - dt.min(axis=0).values) + eps )
    nd = nd.T

    ## Distances
    scd = distances*scMatrix #Same class distance
    dcd = distances*dcMatrix #Different class distance
    ## Normalized distances
    scnd = nd*scMatrix #Same class normalized distance
    dcnd = nd*dcMatrix #Different class normalized distance
    
    ### Lambda computation
    plamb = (1 - scnd) * scMatrix
    lamb = (dcnd + plamb)
    lambs = torch.sum(lamb,axis=1)
    lambs2 = (lambs / (torch.max(lambs) + eps)) ** (1/p)
    lambr = torch.sum(lambs2) / (y_orig.shape[0])

    varsc = torch.var(scnd)
    vardf = torch.var(dcnd)
    omega = (1 - (varsc+vardf))**(1/q)
    
    gamma = torch.sum((torch.sum(scMatrix, axis=1) / k) ** (1/r)) / (y_orig.shape[0])
    
    # return (lambr + gamma + omega)/3
    # return lambr
    return (lambr + gamma)/2

def modified_score(origin, y_orig, target=None, y_targ=None, k=4, ipc=None):
    target = origin if type(target) == type(None) else target
    y_targ = y_orig if type(y_targ) == type(None) else y_targ
    k = target.shape[0] - 2 if k+2 >= target.shape[0] else k
    ipc = k+1 if type(ipc) == type(None) else ipc
    eps = 0.0000001

    distances, indices = nearest_neighbors(origin, target, k=ipc+1)
    distances = distances[:,1:]
    indices = indices[:,1:]

    # class by neighbor
    classes = y_targ[indices]
    yMatrix = y_orig.repeat(ipc,1).T
    scMatrix = (yMatrix == classes)*1 # Same class matrix [1 same class, 0 diff class]
    dcMatrix = (scMatrix)*(-1)+1 # Different class matrix [negation of scMatrix]

    ### Lambda Computation ###

    # Same class distance
    dt = distances.T
    dcnd = distances*dcMatrix
    nn_dc = (dcnd + torch.where(dcnd.eq(0.), float('inf'), 0.)).min(axis=1).values
    nd = dt / (nn_dc + eps)
    nd = nd[:k, :]
    nd = nd / torch.stack((torch.ones_like(nd[0]), nd.max(axis=0).values)).max(axis=0).values # Normalize with max(max_from_row, 1.0)
    nd = nd.T

    scMatrix = scMatrix[:, :k]
    scnd = nd*scMatrix

    scCounts = scMatrix.sum(axis=1)
    scndsum = scnd.sum(axis=1) / (scCounts + eps)
    sclamb = 1 - (scndsum.sum() / (torch.count_nonzero(scCounts) + eps))

    dcnd = dcnd[:, :k]
    dcMatrix = dcMatrix[:, :k]

    # Different class distance
    dcnd = dcnd / (dcnd.max() + eps)
    dcop = -1 if torch.all(dcMatrix == 0) else 0
    dcCounts = dcMatrix.sum(axis=1)
    dcndsum = dcnd.sum(axis=1) / (dcCounts + eps)
    dclamb = dcndsum.sum() / (torch.count_nonzero(dcCounts) + eps)
    dclamb = torch.abs(dclamb + dcop)

    lambr = (sclamb + dclamb) / 2

    ## Omega Calculation ###
    # varsc = torch.var(scnd)
    # vardf = torch.var(dcnd)
    # omega = 1 - (varsc+vardf)
    
    ### Gamma computation
    gamma = torch.sum(torch.sum(scMatrix, axis=1) / k) / (y_orig.shape[0]) if k+2 < target.shape[0] else 1.0
    
    # return (lambr + gamma + omega) / 3
    return (lambr + gamma) / 2

def get_icnn_loss(args, logits, way, qry_labels):
    loss = 0

    if 'cross' in args.losses:
        loss = F.cross_entropy(logits, qry_labels)

    if 'suppicnn' in args.losses:
        supp_labels = torch.arange(0, way, 1/args.shot).type(torch.int).cuda()
        supp_score = modified_score(glob.supp_fts, supp_labels)
        loss += (-torch.log(supp_score))

    if 'queryicnn' in args.losses:
        if args.query_protos:
            proto_labels = torch.arange(0, way).type(torch.int).cuda()
            query_score = modified_score(glob.query_fts, qry_labels, glob.prototypes, proto_labels)
        else:
            supp_labels = torch.arange(0, way, 1/args.shot).type(torch.int).cuda()
            query_score = modified_score(glob.query_fts, qry_labels, glob.supp_fts, supp_labels)
        loss += (-torch.log(query_score))

    if 'fullicnn' in args.losses:
        features = torch.cat((glob.supp_fts, glob.query_fts))
        supp_labels = torch.arange(0, way, 1/args.shot).type(torch.int).cuda()
        labels = torch.cat((supp_labels, qry_labels))
        ipc = args.shot+args.query
        score = modified_score(features, labels, ipc=ipc)

        loss += (-torch.log(score))

    count = len(args.losses.split(','))
    loss = loss / count

    return loss