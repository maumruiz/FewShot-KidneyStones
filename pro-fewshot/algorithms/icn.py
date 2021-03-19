import torch
import numpy as np

from sklearn.neighbors import NearestNeighbors
from algorithms.pca import PCA
# from sklearn.manifold import Isomap
# import umap

class ICN():
    def __init__(self, args):
        self.args = args

    def transform(self, supp_fts, query_fts):
        y = np.arange(0, self.args.way, 1/self.args.shot).astype(int)

        red = PCA(n_components=6).fit(supp_fts)
        emb = red.transform(supp_fts)
        score = self._score(embeddings, y)

        # methods = [PCA, umap.UMAP, Isomap]
        # best = {'score': 0.0, 'embeddings': None, 'reducer': None}

        # for method in methods:
        #     reducer = method(n_components=6).fit(X)
        #     embeddings = reducer.transform(X)
        #     score = self.score(embeddings, y)

        #     if score > best['score']:
        #         best['score'] = score
        #         best['embeddings'] = embeddings
        #         best['reducer'] = reducer

        # supp = best['embeddings']
        # query = best['reducer'].transform(qry)
        # return torch.from_numpy(supp).cuda(), torch.from_numpy(query).cuda()
        return supp_fts, query_fts

    def _score(self, X, y, k=3, p=2, q=2, r=2):
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
            class_prototypes: Prototypes aka mean embeddings for each class
        """
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) #min max scale by feature
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = distances[:,1:]
        indices = indices[:,1:]

        classes = y[indices] # class by neighbord
        yMatrix = np.transpose(np.array([list(y)]*k)) #class matrix
        scMatrix = (yMatrix == classes)*1 #same class matrix [1 same class, 0 diff class]
        dcMatrix = (scMatrix)*(-1)+1 #different class matrix [negation of scMatrix]

        ### Normalizing distances between neighbords
        dt = np.transpose(distances)
        nd = (dt - dt.min(axis=0)) / ( (dt.max(axis=0) - dt.min(axis=0)) +0.001 )
        nd = np.round(np.transpose(nd),2)

        ## Distances
        scd = distances*scMatrix #Same class distance
        dcd = distances*dcMatrix #Different class distance
        ## Normalized distances
        scnd = nd*scMatrix #Same class normalized distance
        dcnd = nd*dcMatrix #Different class normalized distance
        
        ### Lambda computation
        plamb = (1 - scnd)*scMatrix
        lamb = (dcnd + plamb)
        lambs = np.sum(lamb,axis=1)
        lambs2 = np.round(((lambs/max(lambs))**(1/p)),2)
        lambr = round(sum(lambs2)/len(y),2)
        
        gamma = round(sum((np.sum(scMatrix,axis=1)/k)**(1/r))/len(y),2)
        
        return round((lambr + gamma)/2,2)


