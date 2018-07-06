import copy
from deep_rl.network.network_utils import *
from deep_rl.network.network_bodies import *
from generative_playground.gpu_utils import to_gpu


class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body):
        super(ActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())


class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 mask_gen = None,
                 gpu=-1):
        super(CategoricalActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.mask_gen = mask_gen
        #self.to(Config.DEVICE)

    def predict(self, obs, action=None):
        #obs_orig = obs
        try:
            # past actions come in as numpy
            obs = to_gpu(torch.from_numpy(obs))
        except:
            pass # None actions for the first step
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        if self.mask_gen is not None:
            if not self.remember_step:
                # mask_gen is stateful, so need to copy it if don't want to remember the step
                tmp_mask_gen = copy.deepcopy(self.mask_gen)
                mask = tmp_mask_gen(obs)
            else:
                mask =self.mask_gen(obs)
            logits = logits - to_gpu(torch.Tensor(1e6 * (1 - mask)))
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample().to(torch.int64)
        if 75 in action:
            pass
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob, dist.entropy().unsqueeze(-1), v
