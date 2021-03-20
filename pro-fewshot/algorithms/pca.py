import torch

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Center data
        self.mean_ = X.mean(0)
        X -= self.mean_

        # SVD
        u, s, v = torch.linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        u, v = self._svd_flip(u, v)

        self.components_  = v[:self.n_components].t()
        self.explained_variance_ = torch.mul(s[:self.n_components], s[:self.n_components])/(n_samples-1)

        return self

    def transform(self, X):
        X = X - self.mean_
        X_transformed = torch.matmul(X, self.components_)
        return X_transformed

    def _svd_flip(self, u, v, u_based_decision=True):
        """Sign correction to ensure deterministic output from SVD.
        Adjusts the columns of u and the rows of v such that the loadings in the
        columns in u that are largest in absolute value are always positive.
        Parameters
        ----------
        u : ndarray
            u and v are the output of `linalg.svd`, with matching inner
            dimensions so one can compute `np.dot(u * s, v)`.
        v : ndarray
            u and v are the output of `linalg.svd`, with matching inner
            dimensions so one can compute `np.dot(u * s, v)`.
            The input v should really be called vt to be consistent with scipy's
            ouput.
        u_based_decision : bool, default=True
            If True, use the columns of u as the basis for sign flipping.
            Otherwise, use the rows of v. The choice of which variable to base the
            decision on is generally algorithm dependent.
        Returns
        -------
        u_adjusted, v_adjusted : arrays with the same dimensions as the input.
        """
        if u_based_decision:
            # columns of u, rows of v
            max_abs_cols = torch.argmax(torch.abs(u), dim=0)
            signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
            u *= signs
            v *= signs.unsqueeze(-1)
        else:
            # rows of v, columns of u
            max_abs_rows = torch.argmax(torch.abs(v), dim=1)
            signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
            u *= signs
            v *= signs.unsqueeze(-1)
        return u, v