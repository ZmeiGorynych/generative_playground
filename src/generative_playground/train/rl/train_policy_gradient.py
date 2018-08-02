try:
    import generative_playground
except:
    import sys, os, inspect
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../../..')
    #sys.path.append('../../../../DeepRL')
    sys.path.append('../../../../../transformer_pytorch')

#from deep_rl import *
import torch
#from generative_playground.models.problem.rl.network_heads import CategoricalActorCriticNet
#from generative_playground.train.rl.run_iterations import run_iterations
from generative_playground.rdkit_utils.rdkit_utils import num_atoms, num_aromatic_rings, NormalizedScorer
#from generative_playground.models.problem.rl.DeepRL_wrappers import BodyAdapter, MyA2CAgent
from generative_playground.models.model_settings import get_settings
from generative_playground.codec.grammar_mask_gen import GrammarMaskGenerator
from generative_playground.visdom_helper.visdom_helper import Dashboard
from generative_playground.utils.gpu_utils import to_gpu
from generative_playground.train.rl.main_train_policy_gradient import train_policy_gradient

import logging

def reward_length(smiles):
    '''
    A simple reward to encourage larger molecule length
    :param smiles: list of strings
    :return: reward, list of float
    '''
    if not len(smiles):
        return -1 # an empty string is invalid for our purposes
    atoms = num_atoms(smiles)
    return [-1 if num is None else num for num in atoms]

def reward_aromatic_rings(smiles):
    '''
    A simple reward to encourage larger molecule length
    :param smiles: list of strings
    :return: reward, list of float
    '''
    if not len(smiles):
        return -1 # an empty string is invalid for our purposes
    atoms = num_aromatic_rings(smiles)
    return [-1 if num is None else num+0.5 for num in atoms]

batch_size = 40
drop_rate = 0.3
molecules = True
grammar = True
settings = get_settings(molecules, grammar)
max_steps = 277 #settings['max_seq_length']
invalid_value = -3.5
reward_fun = lambda x: reward_aromatic_rings(x)#lambda x: 2.5 + NormalizedScorer(settings['data_path'],invalid_value=invalid_value)(x)#

model, fitter1, fitter2 = train_policy_gradient(molecules,
                                                grammar,
              EPOCHS = 100,
              BATCH_SIZE = batch_size,
                                      reward_fun=reward_fun,
                                      max_steps=max_steps,
              lr = 1e-4,
              drop_rate = drop_rate,
              decoder_type='attention',
              plot_prefix = 'rings ',
              dashboard = 'policy gradient',
                                                save_file='policy_gradient_rings.h5',
                                                smiles_save_file='pg_smiles_rings.h5',
                                                on_policy_loss_type='best',
                                                off_policy_loss_type='mean')
                                                #preload_file='policy_gradient_run.h5')

while True:
    next(fitter1)
    next(fitter2)