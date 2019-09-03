import torch
import torch.optim as optim
from generative_playground.utils.gpu_utils import device

class MCTSRewardProcessor:
    def __init__(self, loss_fun, model, optimizer_factory, batch_size):
        self.loss_fun = loss_fun
        self.model = model
        self.optimizer_factory = optimizer_factory
        self.batch_size = batch_size
        self.rewards = None
        self.log_ps = None
        self.actions = None
        self.params = None
        self.reset()

    def reset(self):
        self.rewards = []
        self.log_ps = []
        self.actions = []
        self.params = set()

    def __call__(self, reward, log_ps, actions, params):
        self.log_ps.append(torch.stack(log_ps))
        self.actions.append(torch.tensor(actions))
        self.rewards.append(reward)
        for p in params:
            self.params.add(p)
        if len(self.rewards) >= self.batch_size:
            self.optimizer_step()
            self.reset()

    def optimizer_step(self):
        opt = self.optimizer_factory(list(self.params))
        loss = self.loss_fun(self.compose_loss_input())
        opt.step(loss)
        del opt
        del loss

    def compose_loss_input(self):
        rewards = torch.tensor(self.rewards, device=device)

        max_len = 0
        for p in self.log_ps:
            max_len = max(max_len, len(p))
        log_ps = torch.zeros(self.batch_size, max_len, device = device, dtype = self.log_ps[0].dtype)
        for ip, p in enumerate(self.log_ps):
            log_ps[ip, :len(p)] = p

        actions = torch.zeros(self.batch_size, max_len, device=device, dtype=self.actions[0].dtype)
        for ip, p in enumerate(self.actions):
            actions[ip, :len(p)] = p

        valid = torch.ones_like(rewards)
        loss_input = {'rewards': rewards,
                      'logp': log_ps,
                      'actions': actions,
                      'info': (None, valid)}
        return loss_input


def optimizer_factory_gen(lr=0.01, grad_clip=5):
    def get_opt(params):
        return OptimizerWrapper(params, lr, grad_clip)
    return get_opt


class OptimizerWrapper:
    def __init__(self, params, lr=0.01, grad_clip=5):
        self.params = params
        self.grad_clip = grad_clip
        self.optimizer = optim.Adam(self.params, lr=lr)

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
