import numpy as np

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
        self.reward_fun = reward_fun
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.actions = []
        self.done_rewards = [None for _ in range(self.batch_size)]
        return [None]*self.batch_size

    def step(self, action):
        '''
        Convention says environment outputs np.arrays
        :param action: LongTensor(batch_size), or np.array(batch_sizelast discrete action chosen
        :return:
        '''
        try: # in case action is a torch.Tensor
            action = action.cpu().to_numpy()
        except:
            pass

        self.actions.append(action[:,None])

        next_state = action
        if len(self.actions) < self._max_episode_steps:
            done = action == self.action_dim-1 # max index is padding, by convention
        else:
            done = np.ones_like(action)

        reward = np.zeros_like(action)
        # for those sequences just computed, calculate the reward
        for i in range(len(action)):
            if self.done_rewards[i] is None and done[i]:
                this_action_seq = np.concatenate(self.actions, axis=1)[i:(i+1),:]
                this_char_seq = self.codec.decode_from_actions(this_action_seq) # codec expects a batch
                #print('this_char_seq:', this_char_seq)
                this_reward = self.reward_fun(this_char_seq)[0]
                self.done_rewards[i] = this_reward
                reward[i] = this_reward

        return next_state, reward, done, None

    def seed(self, random_seed):
        return random_seed