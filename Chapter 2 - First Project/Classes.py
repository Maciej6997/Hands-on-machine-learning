from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

class StandardScalerClone(BaseEstimator, TransformerMixin):
    
    def ___init__(self, with_mean = True):
        self.with_mean = with_mean
    
    def fit(self, X, y = None):
        X = check_array(X)
        self.mean_ = X.mean(axis = 0)
        self.scale_ = X.std(axis = 0)
        self.n_feauteres_in_ = X.shape[0]
        return self
    
    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        assert self.n_feauteres_in_ == X.shape[1]
        if self.with_mean:
            X = X-self.mean_
        return X / self.scale_
    

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_clusters = 10, gamma = 1.0, random_state = None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
    
    def fit(self, X, y = None, sample_weight = None):
        self.k_means_ = KMeans(self.n_clusters, random_state = self.random_state)
        self.k_means_ = self.k_means_.fit(X, sample_weight=sample_weight)
        return self
    
    def transform(self, X):
        return rbf_kernel(X, self.k_means_.cluster_centers_, gamma = self.gamma)
    
    def get_features_names_out(self, names = None):
        return [f"Similarity {i} cluster" for i in range(self.n_clusters)]
