from torch import nn as nn


class ModelForOneStepDecoder(nn.Module):
    def reset_state(self):
        '''
        Sets up the model for next iteration run
        :return:
        '''
        raise NotImplementedError()

    def forward(self, enc_output, last_action=None, last_action_pos=None):
        '''
        One step of the model
        :param enc_output: encoder output, batch of vectors or sequences of vectors
        :param last_action: batch of ints
        :param last_action_pos: int, num of steps since last reset,
        needed eg to do position encoding. We're keeping track of that in the OneStepDecoder already,
        so might as well use it, though could have tracked it in the model itself
        :return: FloatTensor((batch_size x num_actions)), an unmasked vector of logits over next actions
        '''
        # encode last action, if necessary
        # call the model
        # return result
        raise NotImplementedError()