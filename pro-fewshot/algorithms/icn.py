import torch
import numpy as np

from sklearn.neighbors import NearestNeighbors

class ICN():
    def __init__(self, args):
        self.args = args
        self.models = self._get_models(args)

    def _get_models(self, args):
        supp_samples = (args.way * args.shot)
        query_samples = (args.way * args.query)
        task_samples = supp_samples + query_samples

        models = []

        n_components = 6
        n_neighbors = 6

        if n_components >= supp_samples:
            n_components = supp_samples - 2
            n_neighbors = n_components

        if 'pca' in args.icn_models:
            from sklearn.decomposition import PCA
            pca_model = {'model': PCA, 'args': {'n_components': n_components}, 'name':'pca'}
            models.append(pca_model)

        if 'isomap' in args.icn_models:
            from sklearn.manifold import Isomap
            isomap_model = {'model': Isomap, 'args': {'n_components': n_components, 'n_neighbors': n_neighbors}, 'name':'isomap'}
            models.append(isomap_model)
        
        return models

    def transform(self, supp_fts, query_fts):
        X = supp_fts.cpu().detach().numpy()
        y = np.arange(0, self.args.way, 1/self.args.shot).astype(int)
        original_score = self._score(X, y)

        # Initialize ICN scores logger
        if self.args.save_icn_scores:
            self.args.icn_log['original'].append(original_score)
        
        # Initialize best feature reductor
        best = {'score': original_score, 'embeddings': supp_fts, 'reducer': None, 'name': 'original'}

        # Evaluate feature reductor models
        for m in self.models:
            reducer = m['model'](**m['args']).fit(X)
            embeddings = reducer.transform(X)
            score = self._score(embeddings, y)

            if self.args.save_icn_scores:
                self.args.icn_log[m['name']].append(score)

            if score > best['score']:
                best['score'] = score
                best['embeddings'] = embeddings
                best['reducer'] = reducer
                best['n_components'] = m['args']['n_components']
                best['name'] = m['name']

        # Select best model
        if best['reducer']:
            n_components = best['n_components']
            supp_fts[:, :n_components] = torch.Tensor(best['embeddings'])
            supp_fts = supp_fts[:, :n_components]
            qry = best['reducer'].transform(query_fts.cpu().detach().numpy())
            query_fts[:, :n_components] = torch.Tensor(qry)
            query_fts = query_fts[:, :n_components]

        if self.args.save_icn_scores:
                self.args.icn_log['best'].append(best['name'])

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
        a = X - X.min(axis=0)
        b = X.max(axis=0) - X.min(axis=0)
        X = np.divide(a , b, out=np.zeros_like(X), where=b!=0) #min max scale by feature
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


