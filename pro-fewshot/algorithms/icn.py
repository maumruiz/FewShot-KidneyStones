import torch
import numpy as np

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.neighbors import NearestNeighbors

class ICN():
    def __init__(self, args):
        self.args = args
        self.models = self._get_models(args)

    def _get_models(self, args):
        supp_samples = (args.way * args.shot)
        query_samples = (args.way * args.query)

        if self.args.icn_reduction_set == 'all':
            task_samples = supp_samples + query_samples
        elif self.args.icn_reduction_set == 'support':
            task_samples = supp_samples

        models = []
        n_components = 6
        n_neighbors = 6

        if n_components >= task_samples:
            n_components = task_samples - 2
            n_neighbors = n_components

        if 'pca' in args.icn_models:
            from sklearn.decomposition import PCA
            pca_model = {'model': PCA, 'args': {'n_components': n_components}, 'name':'pca', 'n_components': n_components}
            models.append(pca_model)

        if 'isomap' in args.icn_models:
            from sklearn.manifold import Isomap
            isomap_model = {'model': Isomap, 'args': {'n_components': n_components, 'n_neighbors': n_neighbors}, 'name':'isomap', 'n_components': n_components}
            models.append(isomap_model)

        if 'kernel_pca' in args.icn_models:
            from sklearn.decomposition import KernelPCA
            kernelpca_model = {'model': KernelPCA, 'args': {'n_components': n_components, 'kernel': 'rbf'}, 'name':'kernel_pca', 'n_components': n_components}
            models.append(kernelpca_model)

        if 'truncated_svd' in args.icn_models:
            from sklearn.decomposition import TruncatedSVD
            truncatedsvd_model = {'model': TruncatedSVD, 'args': {'n_components': n_components}, 'name':'truncated_svd', 'n_components': n_components}
            models.append(truncatedsvd_model)

        if 'feature_agg' in args.icn_models:
            from sklearn.cluster import FeatureAgglomeration
            featureagg_model = {'model': FeatureAgglomeration, 'args': {'n_clusters': n_components }, 'name':'feature_agg', 'n_components': n_components}
            models.append(featureagg_model)

        if 'fast_ica' in args.icn_models:
            from sklearn.decomposition import FastICA
            fast_ica_model = {'model': FastICA, 'args': {'n_components': n_components }, 'name':'fast_ica', 'n_components': n_components}
            models.append(fast_ica_model)

        if 'nmf' in args.icn_models:
            from sklearn.decomposition import NMF
            nmf_model = {'model': NMF, 'args': {'n_components': n_components, 'init': 'nndsvda' }, 'name':'nmf', 'n_components': n_components}
            models.append(nmf_model)
            
        return models

    @ignore_warnings(category=ConvergenceWarning)
    def transform(self, supp_fts, query_fts):
        # Vectors for support and query
        support = supp_fts.cpu().detach().numpy()
        query = query_fts.cpu().detach().numpy()

        # Set X (data used to generate the feature reduction) and y (support data labels)
        if self.args.icn_reduction_set == 'all':
            X = np.concatenate((support, query))
        elif self.args.icn_reduction_set == 'support' or self.args.icn_reduction_type == 'supervised':
            X = support

        y = np.arange(0, self.args.way, 1/self.args.shot).astype(int)

        # Calculate ICNN score with original feature vector
        original_score = self._score(support, y)

        # Initialize ICN scores logger
        if self.args.save_icn_scores:
            self.args.icn_log['original'].append(original_score)
        
        # Initialize best feature reductor
        best = {'score': original_score, 'embeddings': supp_fts, 'reducer': None, 'name': 'original'}

        # Evaluate feature reductor models
        for m in self.models:
            if self.args.icn_reduction_type == 'unsupervised':
                reducer = m['model'](**m['args']).fit(X)
            elif self.args.icn_reduction_type == 'supervised':
                reducer = m['model'](**m['args']).fit(X, y=y)

            embeddings = reducer.transform(support)
            score = self._score(embeddings, y)

            if self.args.save_icn_scores:
                self.args.icn_log[m['name']].append(score)

            if score > best['score']:
                best['score'] = score
                best['embeddings'] = embeddings
                best['reducer'] = reducer
                best['n_components'] = m['n_components']
                best['name'] = m['name']

        # Select best model
        if best['reducer']:
            n_components = best['n_components']
            supp_fts[:, :n_components] = torch.Tensor(best['embeddings'])
            supp_fts = supp_fts[:, :n_components]
            query_embeddings = best['reducer'].transform(query)
            query_fts[:, :n_components] = torch.Tensor(query_embeddings)
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


