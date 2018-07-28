try:
    import generative_playground
except:
    import sys, os, inspect
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../..')
    sys.path.append('../../../../DeepRL')
    sys.path.append('../../../../transformer_pytorch')

#from deep_rl import *
from deep_rl import Config
import torch
from generative_playground.models.problem.rl.network_heads import CategoricalActorCriticNet
from generative_playground.train.rl.run_iterations import run_iterations
from generative_playground.rdkit_utils.rdkit_utils import num_atoms, NormalizedScorer
from generative_playground.models.problem.rl.DeepRL_wrappers import BodyAdapter, MyA2CAgent
from generative_playground.models.problem.rl.task import SequenceGenerationTask
from generative_playground.models.model_settings import get_settings
from generative_playground.codec.grammar_mask_gen import GrammarMaskGenerator
from generative_playground.visdom_helper.visdom_helper import Dashboard
from generative_playground.utils.gpu_utils import to_gpu


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



batch_size = 100
drop_rate = 0.3
molecules = True
grammar = True
settings = get_settings(molecules, grammar)
max_steps = 50 #settings['max_seq_length']
invalid_value = -7.0

task = SequenceGenerationTask(molecules = molecules,
                              grammar = grammar,
                              reward_fun = NormalizedScorer(settings['data_path'],
                                                            invalid_value=invalid_value),
                              batch_size = batch_size,
                              max_steps=50)

if grammar:
    mask_gen = GrammarMaskGenerator(task.env._max_episode_steps, grammar=settings['grammar'])
else:
    mask_gen = None



def a2c_sequence(name = 'a2c_sequence', task=None, body=None):
    config = Config()
    config.num_workers = batch_size # same thing as batch size
    config.task_fn = lambda: task
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.0007)
    config.network_fn = lambda state_dim, action_dim: \
                            to_gpu(CategoricalActorCriticNet(state_dim,
                                                      action_dim,
                                                      body,
                                                      gpu=0,
                                                      mask_gen=mask_gen))
    #config.policy_fn = SamplePolicy # not used
    config.state_normalizer = lambda x: x
    config.reward_normalizer = lambda x: x
    config.discount = 0.99
    config.use_gae = False #TODO: for now, MUST be false as our RNN network isn't com
    config.gae_tau = 0.97
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 0.5
    config.logger = logging.getLogger()#get_logger(file_name='deep_rl_a2c', skip=True)
    config.logger.info('test')
    config.iteration_log_interval
    config.max_steps = 100000
    dash_name = 'DeepRL'
    visdom = Dashboard(dash_name)
    run_iterations(MyA2CAgent(config), visdom, invalid_value=invalid_value)





#
#
# decoder = SelfAttentionDecoderStep(num_actions=task.env.action_dim,
#                                        max_seq_len=task.env._max_episode_steps,
#                                        drop_rate=drop_rate)

from generative_playground.models.decoder.rnn import SimpleRNNDecoder
decoder = SimpleRNNDecoder(z_size=5,
                               hidden_n=512,
                               feature_len=task.env.action_dim,
                               max_seq_length=task.env._max_episode_steps,  # TODO: WHY???
                               drop_rate=drop_rate,
                               use_last_action=True)

# from transformer.OneStepAttentionDecoder import SelfAttentionDecoderStep
# decoder = SelfAttentionDecoderStep(num_actions=task.env.action_dim,
#                                        max_seq_len=task.env._max_episode_steps,
#                                        drop_rate=drop_rate)


body = BodyAdapter(decoder)

a2c_sequence(task=task, body=body)