import torch

from deep_rl import BaseTask
from models.model_settings import get_settings


class SequenceEnvironment:
    def __init__(self,
                 molecules=True,
                 grammar=True,
                 reward_fun = None,
                 batch_size=1):
        settings = get_settings(molecules, grammar)
        self.action_dim = settings['feature_len']
        self.state_dim = self.action_dim
        self._max_episode_steps = settings['feature_len']
        self.codec = settings['codec']
        self.reward_fun = reward_fun,
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.actions = []
        self.done_rewards = [None for _ in range(self.batch_size)]
        return [None]*self.batch_size

    def step(self, action):
        '''

        :param action: LongTensor(batch_size), last discrete action chosen
        :return:
        '''
        self.actions.append(action)
        next_state = action
        if len(self.actions) < self._max_episode_steps:
            done = action == self.action_dim-1 # max index is padding, by convention
        else:
            done = torch.ones_like(action)

        rewards = torch.zeros_like(action).type(FloatTensor)

        # for those sequences just computed, calculate the reward
        for i in range(action):
            if self.done_rewards[i] is None and done[i]:
                this_action_seq = [action[i] for action in self.actions]
                this_char_seq = self.codec.decode_from_actions([this_action_seq]) # codec expects a batch
                this_reward = self.reward_fun(this_char_seq)
                self.done_rewards[i] = this_reward

    def seed(self, random_seed):
        return random_seed


class SequenceGenerationTask(BaseTask):
    def __init__(self, name='seq_gen',
                 molecules = True,
                 grammar = False,
                 reward_fun = None,
                 batch_size = 1,
                 log_dir=None):
        super().__init__(self)
        self.name = name
        self.env = SequenceEnvironment(molecules,
                                       grammar,
                                       reward_fun=reward_fun,
                                       batch_size=batch_size)
        self.action_dim = self.env.action_dim
        self.state_dim = self.env.state_dim
        self.env = self.set_monitor(self.env, log_dir)