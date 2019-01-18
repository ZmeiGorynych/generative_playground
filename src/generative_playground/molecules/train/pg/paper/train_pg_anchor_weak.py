try:
    import generative_playground
except:
    import sys, os, inspect
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../../../..')
    sys.path.append('../../../../../../transformer_pytorch')

import numpy as np
from generative_playground.molecules.rdkit_utils.rdkit_utils  import num_atoms, num_aromatic_rings, num_aliphatic_rings, NormalizedScorer
#from generative_playground.models.problem.rl.DeepRL_wrappers import BodyAdapter, MyA2CAgent
from generative_playground.molecules.model_settings import get_settings
from generative_playground.molecules.train.pg.main_train_policy_gradient import train_policy_gradient


def reward_length(smiles):
    '''
    A simple reward to encourage larger molecule length
    :param smiles: list of strings
    :return: reward, list of float
    '''
    if not len(smiles):
        return -1 # an empty string is invalid for our purposes
    atoms = num_atoms(smiles)
    return np.array([-1 if num is None else num for num in atoms])

def reward_aromatic_rings(smiles):
    '''
    A simple reward to encourage larger molecule length
    :param smiles: list of strings
    :return: reward, list of float
    '''
    return np.array([-1 if num is None else num+0.5 for num in num_aromatic_rings(smiles)])

def reward_aliphatic_rings(smiles):
    '''
    A simple reward to encourage larger molecule length
    :param smiles: list of strings
    :return: reward, list of float
    '''
    return np.array([-1 if num is None else num for num in num_aliphatic_rings(smiles)])


batch_size = 40
drop_rate = 0.3
molecules = True
grammar = 'new'#True#
settings = get_settings(molecules, grammar)
invalid_value = -3*3.5
scorer = NormalizedScorer(invalid_value=invalid_value, sa_mult=20, sa_thresh=-0.5, normalize_scores=True)
max_steps = 277 #settings['max_seq_length']


reward_fun = lambda x: scorer(x)# - 0.2*np.array([0 if num is None else num for num in num_aromatic_rings(x)])# + reward_aliphatic_rings(x)# + 0.05*reward_aromatic_rings(x)#lambda x: reward_aromatic_rings(x)# #lambda x: reward_aromatic_rings(x)#

model, fitter1, fitter2 = train_policy_gradient(molecules,
                                                grammar,
                                                EPOCHS=100,
                                                BATCH_SIZE=batch_size,
                                                reward_fun_on=reward_fun,
                                                max_steps=max_steps,
                                                lr_off=0.0,
                                                lr_on=1e-4,
                                                drop_rate = drop_rate,
                                                decoder_type='attention',#'random',#
                                                plot_prefix='anchor ',
                                                dashboard='anchor weak sa-0.5',#None,#
                                                save_file='paper/policy_gradient_anchor_weak_sa-0.5.h5',
                                                smiles_save_file='paper/pg_smiles_anchor_weak_sa-0.5.h5',
                                                on_policy_loss_type='best',
                                                off_policy_loss_type='mean',
                                                preload_file='paper/policy_gradient_baseline.h5',
                                                anchor_file='paper/policy_gradient_baseline.h5',
                                                anchor_weight=2e7)
#
while True:
    next(fitter1)
    #next(fitter2)