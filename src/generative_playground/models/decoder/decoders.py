import torch
from torch import nn as nn

from generative_playground.utils.gpu_utils import to_gpu, FloatTensor
from generative_playground.models.decoder.policy import SimplePolicy

# # TODO: merge this and UsingAction, make UsingAction into a bool
# class OneStepDecoder(nn.Module):
#     '''
#     One step of a decoder into a discrete space, suitable for use with autoencoders
#     (so the encoded state is a vector not a sequence)
#     '''
#     def __init__(self, model, max_len=None):
#         '''
#         Base class for doing the differentiable part of one decoding step
#         :param model: a differentiable model used in the steps
#         '''
#         super().__init__()
#         self.n = 0
#         self.model = to_gpu(model)
#         self.model.eval()
#         self.max_len = max_len
#
#
#     def init_decoder_output(self, z):
#         '''
#         Start decoding a new batch
#         :param z: batch_size x num actions or batch_size x max_input_len x num_actions encoded state
#         :return: None
#         '''
#         self.z = z
#         self.z_size = z.size()[-1]
#         #self.n = 0
#         try:
#             self.model.init_encoder_output(z)
#         except:
#             pass
#
#     def forward(self, action):
#         '''
#         # the differentiable part of one decoding step
#         :param action: LongTensor((batch_size)), last discrete action chosen by the policy,
#         None for the very first action choice
#         :return: FloatTensor((batch_size x num_actions)), an unmasked vector of logits over next actions
#         '''
#         #if self.n < self.max_len:
#             #out = self.model(self.z)
#         out = self.model(last_action=action)#,
#                          #last_action_pos=self.n - 1)
#         out = torch.squeeze(out,1)
#         #self.n += 1
#         return out
#         # else:
#         #     raise StopIteration()



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