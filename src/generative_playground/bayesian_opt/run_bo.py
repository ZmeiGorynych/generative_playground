import pickle
import gzip
import h5py
import os, inspect,sys
import numpy as np
from bayesian_opt.get_score_components import get_score_components
from generative_playground.codec import grammar_codec as grammar_model
import GPy
import GPyOpt
my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#sys.path.insert(0, my_location + '/../')
data_dir = my_location + '/../data/'

with h5py.File(data_dir + 'zinc_grammar_latent_and_scores.h5', 'r') as h5f:
    scores = np.array(h5f['scores'])
    latents = np.array(h5f['latent'])

print('aaa')
stds = scores.std(axis=0)
means = scores.mean(axis=0)


def normalized_score(x):
    x_norm = (x -means)/stds
    return x_norm.sum(axis=1)


norm_scores = normalized_score(scores)
grammar_model = grammar_model.ZincGrammarModel()


def latent_to_score(latent):
    smiles = grammar_model.decode(latent, validate = True, max_attempts=5)
    pre_scores = np.array([get_score_components(s) for s in smiles])
    return normalized_score(pre_scores)


domain = [(-4,4) for _ in range(latents.shape[1])]
myProblem = GPyOpt.methods.BayesianOptimization(latent_to_score,
                                                domain = [{'name': 'var_1', 'type': 'continuous', 'domain': domain}],
                                                model_type ='sparseGP',
                                                X = latents,
                                                y = norm_scores)
#gpyopt goes here

# random_seed = int(np.loadtxt('../random_seed.txt'))
# np.random.seed(random_seed)
#
# # We load the data
#
# X = np.loadtxt('../../latent_features_and_targets_grammar/latent_faetures.txt')
# y = -np.loadtxt('../../latent_features_and_targets_grammar/targets.txt')
# y = y.reshape((-1, 1))
#
# n = X.shape[ 0 ]
# permutation = np.random.choice(n, n, replace = False)
#
# X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
# X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]
#
# y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
# y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]
#
# np.random.seed(random_seed)
#
# iteration = 0
# while iteration < 5:
#
#     # We fit the GP
#
#     np.random.seed(iteration * random_seed)
#     M = 500
#     sgp = SparseGP(X_train, 0 * X_train, y_train, M)
#     sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0,  \
#         y_test, minibatch_size = 10 * M, max_iterations = 50, learning_rate = 0.0005)
#
#     pred, uncert = sgp.predict(X_test, 0 * X_test)
#     error = np.sqrt(np.mean((pred - y_test)**2))
#     testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
#     print 'Test RMSE: ', error
#     print 'Test ll: ', testll
#
#     pred, uncert = sgp.predict(X_train, 0 * X_train)
#     error = np.sqrt(np.mean((pred - y_train)**2))
#     trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
#     print 'Train RMSE: ', error
#     print 'Train ll: ', trainll
#
#     # We load the decoder to obtain the molecules
#
#     import sys
#     sys.path.insert(0, '../../../')
#     grammar_weights = '../../../pretrained/zinc_vae_grammar_L56_E100_val.hdf5'
#     grammar_model = grammar_model.ZincGrammarModel(grammar_weights)
#
#     # We pick the next 50 inputs
#
#     next_inputs = sgp.batched_greedy_ei(50, np.min(X_train, 0), np.max(X_train, 0))
#
#     valid_smiles_final = decode_from_latent_space(next_inputs, grammar_model)
#
#     from rdkit.Chem import Descriptors
#     from rdkit.Chem import MolFromSmiles
#
#     new_features = next_inputs
#
#     save_object(valid_smiles_final, "results/valid_smiles{}.dat".format(iteration))
#
#     logP_values = np.loadtxt('../../latent_features_and_targets_grammar/logP_values.txt')
#     SA_scores = np.loadtxt('../../latent_features_and_targets_grammar/SA_scores.txt')
#     cycle_scores = np.loadtxt('../../latent_features_and_targets_grammar/cycle_scores.txt')
#     SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
#     logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
#     cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)
#
#     targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized
#
#     from data_utils import sascorer
#     import networkx as nx
#     from rdkit.Chem import rdmolops
#
#     scores = []
#     for i in range(len(valid_smiles_final)):
#         if valid_smiles_final[ i ] is not None:
#             current_log_P_value = Descriptors.MolLogP(MolFromSmiles(valid_smiles_final[ i ]))
#             current_SA_score = -sascorer.calculateScore(MolFromSmiles(valid_smiles_final[ i]))
#             cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(valid_smiles_final[ i ]))))
#             if len(cycle_list) == 0:
#                 cycle_length = 0
#             else:
#                 cycle_length = max([ len(j) for j in cycle_list ])
#             if cycle_length <= 6:
#                 cycle_length = 0
#             else:
#                 cycle_length = cycle_length - 6
#
#             current_cycle_score = -cycle_length
#
#             current_SA_score_normalized = (current_SA_score - np.mean(SA_scores)) / np.std(SA_scores)
#             current_log_P_value_normalized = (current_log_P_value - np.mean(logP_values)) / np.std(logP_values)
#             current_cycle_score_normalized = (current_cycle_score - np.mean(cycle_scores)) / np.std(cycle_scores)
#
#             score = (current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized)
#         else:
#             score = -max(y)[ 0 ]
#
#         scores.append(-score)
#         print(i)
#
#     print(valid_smiles_final)
#     print(scores)
#
#     save_object(scores, "results/scores{}.dat".format(iteration))
#
#     if len(new_features) > 0:
#         X_train = np.concatenate([ X_train, new_features ], 0)
#         y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)
#
#     iteration += 1
#
#     print(iteration)
