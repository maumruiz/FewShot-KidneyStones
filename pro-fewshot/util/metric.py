import torch
import numpy as np

def euclidean_dist(x, y):
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    assert d == y.shape[1]

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    # logits = -((x - y)**2).sum(dim=2)
    dist = torch.pow(x - y, 2).sum(2)
    return dist

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()