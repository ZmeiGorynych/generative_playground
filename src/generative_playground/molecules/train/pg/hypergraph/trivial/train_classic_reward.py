try:
    import generative_playground
except:
    import sys, os, inspect

    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../../../../../..')
    # sys.path.append('../../../../DeepRL')
    # sys.path.append('../../../../../transformer_pytorch')
import numpy as np
# from deep_rl import *
# from generative_playground.models.problem.rl.network_heads import CategoricalActorCriticNet
# from generative_playground.train.rl.run_iterations import run_iterations
from generative_playground.molecules.rdkit_utils.rdkit_utils import num_atoms, num_aromatic_rings, NormalizedScorer
# from generative_playground.models.problem.rl.DeepRL_wrappers import BodyAdapter, MyA2CAgent
from generative_playground.molecules.model_settings import get_settings
from generative_playground.molecules.train.pg.hypergraph.main_train_policy_gradient_minimal import train_policy_gradient
from generative_playground.codec.hypergraph_grammar import GrammarInitializer



batch_size = 30 # 20
drop_rate = 0.5
molecules = True
grammar_cache = 'hyper_grammar_guac_10k_with_clique_collapse.pickle'#'hyper_grammar.pickle'
grammar = 'hypergraph:' + grammar_cache
# settings = get_settings(molecules, grammar)
# max_steps = 277  # settings['max_seq_length']
invalid_value = -3.5
scorer = NormalizedScorer(invalid_value=invalid_value, normalize_scores=True)
reward_fun = lambda x: np.tanh(0.1*scorer(x)) # lambda x: reward_aromatic_rings(x)#
# later will run this ahead of time
# gi = GrammarInitializer(grammar_cache)

max_steps = 45
root_name = 'classic_logP'
model, gen_fitter, disc_fitter = train_policy_gradient(molecules,
                                                       grammar,
                                                       EPOCHS=100,
                                                       BATCH_SIZE=batch_size,
                                                       reward_fun_on=reward_fun,
                                                       max_steps=max_steps,
                                                       lr_on=3e-5,
                                                       lr_discrim=5e-4,
                                                       discrim_wt=0.1,
                                                       p_thresh=-10,
                                                       drop_rate=drop_rate,
                                                       reward_sm=0.5,
                                                       decoder_type='attn_graph',  # 'attention',
                                                       plot_prefix='',
                                                       dashboard=root_name,  # 'policy gradient',
                                                       save_file_root_name=root_name,
                                                       smiles_save_file=root_name + '_smiles',
                                                       on_policy_loss_type='advantage_record',
                                                       half_float=True)
# preload_file='policy_gradient_run.h5')

while True:
    next(gen_fitter)
    for _ in range(1):
        next(disc_fitter)
