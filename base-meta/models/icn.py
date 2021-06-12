import torch
import numpy as np

from .models import register
import utils.settings as settings
import utils.few_shot as fs

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import NearestNeighbors

@register('icn')
class ICN():
    def __init__(self):
        self.args = settings.icn
        self.models = self._get_models()
        self.score = self._original_score if self.args['icn_original_score'] else self._score

    def _get_models(self):
        supp_samples = (settings.way * settings.shot)
        query_samples = (settings.way * settings.query)

        if self.args['icn_reduction_set'] == 'all':
            task_samples = supp_samples + query_samples
        elif self.args['icn_reduction_set'] == 'support':
            task_samples = supp_samples

        hdim = 512


        components_list = []
        if self.args['icn_multiple_components']:
            for _ in range(self.args['icn_n_dims']):
                hdim = hdim // 2
                components_list.append(hdim)
        else:
            components_list.append(self.args['icn_n_dims'])

        models = []

        if 'pca' in self.args['icn_models']:
            from sklearn.decomposition import PCA
            for n_components in components_list:
                n_components = task_samples - 2 if n_components >= task_samples else n_components
                pca_model = {'model': PCA, 'args': {'n_components': n_components}, 'name':'pca', 'n_components': n_components}
                models.append(pca_model)

        if 'isomap' in self.args['icn_models']:
            from sklearn.manifold import Isomap
            for n_components in components_list:
                n_components = task_samples - 2 if n_components >= task_samples else n_components
                n_neighbors = n_components if n_components < 5 else 5
                isomap_model = {'model': Isomap, 'args': {'n_components': n_components, 'n_neighbors': n_neighbors}, 'name':'isomap', 'n_components': n_components}
                models.append(isomap_model)

        if 'umap' in self.args['icn_models']:
            import umap
            for n_components in components_list:
                n_components = task_samples - 2 if n_components >= task_samples else n_components
                umap_model = {'model': umap.UMAP, 'args': {'n_components': n_components}, 'name':'umap', 'n_components': n_components}
                models.append(umap_model)

        if 'kernel_pca' in self.args['icn_models']:
            from sklearn.decomposition import KernelPCA
            for n_components in components_list:
                n_components = task_samples - 2 if n_components >= task_samples else n_components
                kernelpca_model = {'model': KernelPCA, 'args': {'n_components': n_components, 'kernel': 'rbf'}, 'name':'kernel_pca', 'n_components': n_components}
                models.append(kernelpca_model)

        if 'truncated_svd' in self.args['icn_models']:
            from sklearn.decomposition import TruncatedSVD
            for n_components in components_list:
                n_components = task_samples - 2 if n_components >= task_samples else n_components
                truncatedsvd_model = {'model': TruncatedSVD, 'args': {'n_components': n_components}, 'name':'truncated_svd', 'n_components': n_components}
                models.append(truncatedsvd_model)

        if 'feature_agg' in self.args['icn_models']:
            from sklearn.cluster import FeatureAgglomeration
            for n_components in components_list:
                n_components = task_samples - 2 if n_components >= task_samples else n_components
                featureagg_model = {'model': FeatureAgglomeration, 'args': {'n_clusters': n_components }, 'name':'feature_agg', 'n_components': n_components}
                models.append(featureagg_model)

        if 'fast_ica' in self.args['icn_models']:
            from sklearn.decomposition import FastICA
            for n_components in components_list:
                n_components = task_samples - 2 if n_components >= task_samples else n_components
                fast_ica_model = {'model': FastICA, 'args': {'n_components': n_components }, 'name':'fast_ica', 'n_components': n_components}
                models.append(fast_ica_model)

        if 'nmf' in self.args['icn_models']:
            from sklearn.decomposition import NMF
            for n_components in components_list:
                n_components = task_samples - 2 if n_components >= task_samples else n_components
                nmf_model = {'model': NMF, 'args': {'n_components': n_components, 'init': 'nndsvda' }, 'name':'nmf', 'n_components': n_components}
                models.append(nmf_model)
            
        return models

    @ignore_warnings(category=ConvergenceWarning)
    def transform(self, supp_fts, query_fts):
        n_batch = supp_fts.shape[0]

        # Vectors for support and query
        support = supp_fts.cpu().detach().numpy()
        query = query_fts.cpu().detach().numpy()

        support = support.reshape(-1, support.shape[-1])
        query = query.reshape(-1, query.shape[-1])

        # Set X (data used to generate the feature reduction) and y (support data labels)
        if self.args['icn_reduction_set'] == 'all':
            X = np.concatenate((support, query))
        elif self.args['icn_reduction_set'] == 'support' or self.args['icn_reduction_type'] == 'supervised':
            X = support

        # y = np.arange(0, settings.way, 1/settings.shot).astype(int)
        y = fs.make_nk_label(settings.way, settings.shot, ep_per_batch=n_batch).numpy()

        # Calculate ICNN score with original feature vector
        original_score = self.score(support, y)

        # Initialize ICN scores logger
        if self.args['save_icn_scores']:
            self.args['icn_log']['original'].append(original_score)
        
        # Initialize best feature reductor
        best = {'score': original_score, 'embeddings': supp_fts, 'reducer': None, 'name': 'original'}

        # Evaluate feature reductor models
        for m in self.models:
            if self.args['icn_reduction_type'] == 'unsupervised':
                reducer = m['model'](**m['args']).fit(X)
            elif self.args['icn_reduction_type'] == 'supervised':
                reducer = m['model'](**m['args']).fit(X, y=y)

            embeddings = reducer.transform(support)
            score = self.score(embeddings, y)

            if self.args['save_icn_scores']:
                if f'{m["name"]}_{m["n_components"]}dims' not in self.args['icn_log']:
                    self.args['icn_log'][f'{m["name"]}_{m["n_components"]}dims'] = []
                self.args['icn_log'][f'{m["name"]}_{m["n_components"]}dims'].append(score)

            if score > best['score']:
                best['score'] = score
                best['embeddings'] = embeddings
                best['reducer'] = reducer
                best['n_components'] = m['n_components']
                best['name'] = m['name']

        # Select best model
        if best['reducer']:
            supp_fts = supp_fts.reshape(-1, supp_fts.shape[-1])
            query_fts = query_fts.reshape(-1, query_fts.shape[-1])
            n_components = best['n_components']
            supp_fts[:, :n_components] = torch.Tensor(best['embeddings'])
            supp_fts = supp_fts[:, :n_components]
            query_embeddings = best['reducer'].transform(query)
            query_fts[:, :n_components] = torch.Tensor(query_embeddings)
            query_fts = query_fts[:, :n_components]
            supp_fts = supp_fts.reshape(n_batch, settings.way, settings.shot, n_components)
            query_fts = query_fts.reshape(n_batch, settings.way*settings.query, n_components)

        if self.args['save_icn_scores']:
                self.args['icn_log']['best'].append(best['name'])

        return supp_fts, query_fts

    def _original_score(self, X, y, k=3, p=2, q=2, r=2):
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

    def _score(self, X, y, k=4, p=1, q=1, r=1):
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
        if k+2 > X.shape[0]:
            k = X.shape[0] - 2
            
        nbrs = NearestNeighbors(n_neighbors=k+2).fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = distances[:,1:]
        indices = indices[:,1:]

        classes = y[indices] # Class by neighbor
        yMatrix = np.transpose(np.array([list(y)]*(k+1))) # Class matrix
        scMatrix = (yMatrix == classes)*1 # Same class matrix [1 same class, 0 diff class]
        dcMatrix = (scMatrix)*(-1)+1 # Different class matrix [negation of scMatrix]

        ### Lambda Computation

        # Same class distance
        dt = np.transpose(distances)
        dcnd = distances*dcMatrix
        nn_dc = np.ma.masked_equal(dcnd, 0.0, copy=False).min(axis=1).data
        nd = (dt / (nn_dc + 0.00001 ) )
        nd = nd / nd.max(axis=0)
        nd = np.transpose(nd)

        nd = nd[:, :-1]
        scMatrix = scMatrix[:, :-1]
        scnd = nd*scMatrix

        scndsum = np.sum(scnd, axis=1) / (scMatrix.sum(axis=1)+0.000001)
        sclamb = 1 - (np.sum(scndsum) / (np.count_nonzero(scndsum)+0.000001))

        dcnd = dcnd[:, :-1]
        dcMatrix = dcMatrix[:, :-1]

        # Different class distance
        dcnd = dcnd / (dcnd.max() + 0.000001)
        dcop = -1 if (dcMatrix == 0).all() else 0
        dcndsum = np.sum(dcnd, axis=1) / (dcMatrix.sum(axis=1)+0.000001)
        dclamb = np.sum(dcndsum) / (np.count_nonzero(dcndsum)+0.000001)
        dclamb = np.abs(dclamb + dcop)

        lambr = (sclamb + dclamb) / 2

        ## Omega Calculation
        varsc = np.var(scnd)
        vardf = np.var(dcnd)
        omega = np.round((1 - (varsc+vardf))**(1/q),2)
        
        ### Gamma computation
        if (scMatrix == 0).all() or (scMatrix == 1).all():
            gamma = 1.0
        else:
            gamma = sum((np.sum(scMatrix,axis=1)/k)**(1/r))/len(y)

        # print(f"lambda: {lambr}, omega: {omega}, gamma: {gamma}")
        
        return (lambr + gamma + omega)/3

