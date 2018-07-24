import torch
from torch import nn as nn
from torch import functional as F

from generative_playground.utils.gpu_utils import to_gpu, FloatTensor
from generative_playground.models.decoder.policy import SimplePolicy



class OneStepDecoderContinuous(nn.Module):
    def __init__(self,model):
        '''
        Wrapper for a continuous decoder that doesn't look at last action chosen, eg simple RNN
        :param model:
        '''
        super().__init__()
        self.model = to_gpu(model)
        self.model.eval()

    def init_encoder_output(self, z):
        self.n = 0
        self.z = z
        self.z_size = z.size()[-1]
        try:
            self.model.init_encoder_output(z)
        except:
            pass
        self.logits = self.model.forward(z)

    def forward(self, action=None):
        '''
        Gets the output sequence all at once, then feeds it back one step at a time
        :param action: ignored
        :return: a vector of logits over next actions
        '''
        if self.n < self.logits.shape[1]:
            out = torch.squeeze(self.logits[:, self.n, :],1)
            self.n +=1
            return out
        else:
            raise StopIteration()

class SimpleDiscreteDecoder(nn.Module):
    def __init__(self, stepper, policy: SimplePolicy, mask_gen = None, bypass_actions=False):
        '''
        A simple discrete decoder, alternating getting logits from model and actions from policy
        :param stepper:
        :param policy: choose an action from the logits, can be max, or random sample,
        or choose from pre-determined target sequence. Only depends on current logits + history,
        can't handle multi-step strategies like beam search
        :param mask_fun: takes in one-hot encoding of previous action (for now that's all we care about)
        '''
        super().__init__()
        self.stepper = to_gpu(stepper)
        #self.z_size = self.stepper.z_size
        self.policy = policy
        self.mask_gen = mask_gen
        self.bypass_actions = bypass_actions

    def forward(self, z):
        # initialize the decoding model
        self.stepper.init_encoder_output(z)
        if self.bypass_actions:
            return None, self.stepper.logits
        out_logits = []
        out_actions = []
        last_action = [None]*len(z)
        step = 0
        # as it's PyTorch, can determine max_len dynamically, by when the stepper raises StopIteration
        while True:
            try:
  #          if True:
                #  batch x num_actions
                next_logits = self.stepper(last_action)
                # just in case we were returned a sequence of length 1
                next_logits = torch.squeeze(next_logits, 1)
                if self.mask_gen is not None:
                    # mask_gen might return a numpy mask
                    mask = FloatTensor(self.mask_gen(last_action))
                    masked_logits = next_logits - 1e4*(1-mask)
                else:
                    masked_logits = next_logits

                next_action = self.policy(masked_logits)
                out_logits.append(torch.unsqueeze(masked_logits,1))
                out_actions.append(torch.unsqueeze(next_action,1))
                last_action = next_action
            except StopIteration as e:
                #print(e)
                break
        if self.mask_gen is not None:
            self.mask_gen.reset()
        out_actions_all = torch.cat(out_actions, 1)
        out_logits_all = torch.cat(out_logits, 1)
        return out_actions_all, out_logits_all

class SimpleDiscreteDecoderWithEvnv(nn.Module):
    def __init__(self,
                 stepper,
                 policy: SimplePolicy,
                 mask_gen=None,
                 task=None):
        '''
        A simple discrete decoder, alternating getting logits from model and actions from policy
        :param stepper:
        :param policy: choose an action from the logits, can be max, or random sample,
        or choose from pre-determined target sequence. Only depends on current logits + history,
        can't handle multi-step strategies like beam search
        :param mask_fun: takes in one-hot encoding of previous action (for now that's all we care about)
        :param task: environment that returns rewards and whether the episode is finished
        '''
        super().__init__()
        self.stepper = to_gpu(stepper)
        #self.z_size = self.stepper.z_size
        self.policy = policy
        self.mask_gen = mask_gen
        self.task = task

    def forward(self, z=None):
        # initialize the decoding model
        self.stepper.init_encoder_output(z)
        if self.bypass_actions:
            return None, self.stepper.logits
        out_logits = []
        out_actions = []
        out_rewards = []
        out_terminals = []
        if z is not None:
            last_state = [None]*len(z)
        elif self.task is not None:
            last_state = self.task.reset()
        step = 0
        # as it's PyTorch, can determine max_len dynamically, by when the stepper raises StopIteration
        while True:
            try:
  #          if True:
                #  batch x num_actions
                next_logits = self.stepper(last_state)
                # just in case we were returned a sequence of length 1
                next_logits = torch.squeeze(next_logits, 1)
                if self.mask_gen is not None:
                    # mask_gen might return a numpy mask
                    mask = FloatTensor(self.mask_gen(last_state))
                    masked_logits = next_logits - 1e4*(1-mask)
                else:
                    masked_logits = next_logits

                next_action = self.policy(masked_logits)
                out_logits.append(torch.unsqueeze(masked_logits,1))
                out_actions.append(torch.unsqueeze(next_action,1))
                if self.task is None:
                    last_state = next_action
                else:
                    # TODO does that play nicely after sequence end?
                    last_state, rewards, terminals, _ = self.task.step(next_action.detach().cpu().numpy())
                    out_rewards.append(rewards)
                    out_terminals.append(terminals)
            except StopIteration as e:
                #print(e)
                break
        if self.mask_gen is not None:
            self.mask_gen.reset()
        out_actions_all = torch.cat(out_actions, 1)
        out_logits_all = torch.cat(out_logits, 1)
        return out_actions_all, out_logits_all

class PolicyGradientLoss(nn.Module):
    def forward(self, logits, actions, rewards, terminals):
        '''
        Calculate a policy gradient loss given the logits
        :param logits: batch x seq_length x num_actions floats
        :param actions: batch x seq_length ints
        :param rewards: batch x seq_len floats
        :param terminals: batch x seq_len x num_action True if
        :return:
        '''
        log_p = torch.nn.LogSoftmax(logits, dim=2)
        total_rewards = rewards.sum(1)
        term_mask = ...
        total_loss = 0
        masked_logp = log_p * term_mask
        for i in range(logits.size()[1]):
            dloss = -masked_logp[:, i, actions[:,i]].mean()
            total_loss += dloss

        total_loss *= total_rewards
        return total_loss
