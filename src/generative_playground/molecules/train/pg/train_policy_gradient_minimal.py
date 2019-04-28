try:
    import generative_playground
except:
    import sys, os, inspect

    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../../../..')
    print('../../../..')
    # sys.path.append('../../../../DeepRL')
    # sys.path.append('../../../../../transformer_pytorch')

import os
# from deep_rl import *
# from generative_playground.models.problem.rl.network_heads import CategoricalActorCriticNet
# from generative_playground.train.rl.run_iterations import run_iterations
from generative_playground.molecules.rdkit_utils.rdkit_utils import num_atoms, num_aromatic_rings, NormalizedScorer
# from generative_playground.models.problem.rl.DeepRL_wrappers import BodyAdapter, MyA2CAgent
from generative_playground.molecules.model_settings import get_settings
from generative_playground.molecules.train.pg.main_train_policy_gradient_minimal import train_policy_gradient
from generative_playground.codec.hypergraph_grammar import GrammarInitializer


def reward_length(smiles):
    '''
    A simple reward to encourage larger molecule length
    :param smiles: list of strings
    :return: reward, list of float
    '''
    if not len(smiles):
        return -1  # an empty string is invalid for our purposes
    atoms = num_atoms(smiles)
    return [-1 if num is None else num for num in atoms]


def reward_aromatic_rings(smiles):
    '''
    A simple reward to encourage larger molecule length
    :param smiles: list of strings
    :return: reward, list of float
    '''
    if not len(smiles):
        return -1  # an empty string is invalid for our purposes
    atoms = num_aromatic_rings(smiles)
    return [-1 if num is None else num + 0.5 for num in atoms]


batch_size = 30#20
drop_rate = 0.0#0.3
molecules = True
grammar_cache = 'hyper_grammar.pickle'
grammar = 'hypergraph:' + grammar_cache
settings = get_settings(molecules, grammar)
# max_steps = 277  # settings['max_seq_length']
invalid_value = -3.5
scorer = NormalizedScorer(invalid_value=invalid_value)
reward_fun = lambda x: 2.5 + scorer(x)  # lambda x: reward_aromatic_rings(x)#
# later will run this ahead of time
gi = GrammarInitializer(grammar_cache)
gf = gi.full_grammar_filename(grammar_cache)
go = gi.full_own_filename(grammar_cache)
# if os.path.isfile(gf):
#     os.remove(gf)
# if os.path.isfile(go):
#     os.remove(go)

if os.path.isfile(go):
    gi = GrammarInitializer.load(go)
    max_steps = gi.max_len + 1
else:
    max_steps = gi.init_grammar(100)
    # TODO: remove this l
max_steps = 30
model, fitter1, fitter2 = train_policy_gradient(molecules,
                                       grammar,
                                       EPOCHS=100,
                                       BATCH_SIZE=batch_size,
                                       reward_fun_on=reward_fun,
                                       max_steps=max_steps,
                                       lr_on=1e-4,
                                       drop_rate=drop_rate,
                                       decoder_type='attn_graph',#'attention',
                                       plot_prefix='hg ',
                                       dashboard='hypergraph',  # 'policy gradient',
                                       save_file='policy_gradient_hg1.h5',
                                       smiles_save_file=None,#'pg_smiles_hg1.h5',
                                       on_policy_loss_type='best',
                                       off_policy_loss_type='mean')
# preload_file='policy_gradient_run.h5')

while True:
    next(fitter1)
    for _ in range(1):
        next(fitter2)
