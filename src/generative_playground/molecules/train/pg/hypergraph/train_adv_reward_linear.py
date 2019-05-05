try:
    import generative_playground
except:
    import sys, os, inspect

    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../../../../..')
    # sys.path.append('../../../../DeepRL')
    # sys.path.append('../../../../../transformer_pytorch')

# from deep_rl import *
# from generative_playground.models.problem.rl.network_heads import CategoricalActorCriticNet
# from generative_playground.train.rl.run_iterations import run_iterations
from generative_playground.molecules.rdkit_utils.rdkit_utils import num_atoms, num_aromatic_rings, NormalizedScorer
# from generative_playground.models.problem.rl.DeepRL_wrappers import BodyAdapter, MyA2CAgent
from generative_playground.molecules.model_settings import get_settings
from generative_playground.molecules.train.pg.hypergraph.main_train_policy_gradient_minimal import train_policy_gradient
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


batch_size = 20 # 20
drop_rate = 0.5
molecules = True
grammar_cache = 'hyper_grammar.pickle'
grammar = 'hypergraph:' + grammar_cache
settings = get_settings(molecules, grammar)
# max_steps = 277  # settings['max_seq_length']
invalid_value = -3.5
scorer = NormalizedScorer(invalid_value=invalid_value)
reward_fun = scorer #lambda x: np.ones(len(x)) # lambda x: reward_aromatic_rings(x)#
# later will run this ahead of time
gi = GrammarInitializer(grammar_cache)
# if True:
#     gi.delete_cache()
#     gi = GrammarInitializer(grammar_cache)
#     max_steps_smiles = gi.init_grammar(1000)

max_steps = 30
model, gen_fitter, disc_fitter = train_policy_gradient(molecules,
                                                grammar,
                                                EPOCHS=100,
                                                BATCH_SIZE=batch_size,
                                                reward_fun_on=reward_fun,
                                                max_steps=max_steps,
                                                lr_on=3e-5,
                                                lr_discrim=5e-4,
                                                       p_thresh=0.2,
                                                drop_rate=drop_rate,
                                                       reward_sm=0.5,
                                                decoder_type='attn_graph',  # 'attention',
                                                plot_prefix='hg ',
                                                dashboard='true_reward_linear_faster',  # 'policy gradient',
                                                save_file='adv_true_reward_linear_faster.h5',
                                                smiles_save_file=None,  # 'pg_smiles_hg1.h5',
                                                on_policy_loss_type='advantage',#''best',
                                                off_policy_loss_type='mean')
# preload_file='policy_gradient_run.h5')

while True:
    next(gen_fitter)
    for _ in range(1):
        next(disc_fitter)
