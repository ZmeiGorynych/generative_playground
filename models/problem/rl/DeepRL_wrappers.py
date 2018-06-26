from torch import nn
import torch
from gpu_utils import to_gpu
from deep_rl.agent.A2C_agent import A2CAgent

class BodyAdapter(nn.Module):
    '''
    Converts from output_shape convention we use to DeepRL feature_dim convention
    '''
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.feature_dim = model.output_shape[-1]

    def forward(self, x):
        # if the internal model returns a sequence of one element, squeeze that out
        return self.model(x).squeeze(1)

class MyA2CAgent(A2CAgent):
    def __init__(self, config):
        '''
        A new agent gets spawned for every new sequence, reusing the same network
        So this is where we need to init_encoder_output for the network,
        so it knows a new sequence has started
        :param config: the DeepRL config object
        '''
        super().__init__(config)
        try:
            self.dummy_enc_output = to_gpu(torch.zeros(config.num_workers,5)) # 5 just because :)
            self.network.network.phi_body.model.init_encoder_output(self.dummy_enc_output)
        except:
            pass

    def iteration(self):
        # iterate until max_len is reached, through the whole game sequence
        while True:
            try:
                super().iteration()
            except StopIteration:
                # reset the stateful network
                self.network.network.phi_body.model.init_encoder_output(self.dummy_enc_output)
                self.last_episode_rewards = self.episode_rewards
                self.reset_state()
                break

