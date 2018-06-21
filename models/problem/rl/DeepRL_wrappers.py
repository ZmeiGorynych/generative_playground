from torch import nn
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
        return self.model(x)

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
            dummy_enc_output = torch.zeros()# TODO what shape???)
            self.network.phi_body.init_encoder_output(dummy_enc_output)
        except:
            pass