try:
    import generative_playground
except:
    import sys, os, inspect
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../..')
    sys.path.append('../../../../DeepRL')
    sys.path.append('../../../../transformer_pytorch')

#from deep_rl import *
import torch
#from generative_playground.models.problem.rl.network_heads import CategoricalActorCriticNet
#from generative_playground.train.rl.run_iterations import run_iterations
from generative_playground.rdkit_utils.rdkit_utils import num_atoms, NormalizedScorer
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



batch_size = 40
drop_rate = 0.3
molecules = True
grammar = True
settings = get_settings(molecules, grammar)
max_steps = 277 #settings['max_seq_length']
invalid_value = -7.0
reward_fun = reward_length#NormalizedScorer(settings['data_path'],invalid_value=invalid_value)

model, fitter = train_policy_gradient(molecules,
              grammar,
              EPOCHS = 100,
              BATCH_SIZE = batch_size,
                                      reward_fun=reward_fun,
                                      max_steps=max_steps,
              lr = 2e-4,
              drop_rate = drop_rate,
              decoder_type='attention',
              plot_prefix = '',
              dashboard = 'policy gradient',
                                      save_file='dummy.h5',
                                      smiles_save_file='pg_smiles.h5')

while True:
    next(fitter)