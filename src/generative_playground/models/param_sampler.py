import numpy as np
import gzip, pickle
from scipy.stats import percentileofscore
from sklearn import decomposition

def sample_params(x, shape):
    return np.zeros(*shape)

def extract_params_rewards(fn):
    with gzip.open(fn, 'rb') as f:
        graph = pickle.load(f)

    ordered_nodes = list(graph.nodes)
    med_rewards = np.array(
        [np.median(graph.node[n]['best_rewards']) for n in ordered_nodes if 'best_rewards' in graph.node[n]])
    params = np.array([graph.node[n]['params'] for n in ordered_nodes if 'params' in graph.node[n]])
    return params, med_rewards

def values_to_percentiles(rewards, pct_thresh):
    rewards_sorted = sorted(rewards)
    norm_rewards = np.array([percentileofscore(rewards_sorted, x) for x in rewards_sorted])
    return norm_rewards

class ParameterSampler:
    def __init__(self, params, rewards, init_thresh=50, pca_dim=10):
        self.norm_rewards = values_to_percentiles(rewards, init_thresh)
        filt_params = params[self.norm_rewards>init_thresh]
        self.param_std = filt_params.std(axis=0)
        self.pca = decomposition.PCA(n_components=pca_dim)
        self.pca.fit(filt_params)

    def sample(self):
        # now try to approximate sampling from that
        naive_diagonal_sample = self.param_std * np.random.randn(len(self.param_std))
        # and a hugely collinear sample from the PCA
        PCA_pre_sample = np.sqrt(self.pca.explained_variance_) * np.random.randn(1, len(self.pca.components_))
        PCA_sample = PCA_pre_sample.dot(self.pca.components_)
        sample = self.pca.mean_ + naive_diagonal_sample + PCA_sample
        return sample
