''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import generative_playground.models.transformer.Constants as Constants
from generative_playground.models.transformer.DecoderLayerStep import DecoderLayerStep
from generative_playground.utils.gpu_utils import to_gpu, FloatTensor, LongTensor, ByteTensor

__author__ = "Yu-Hsiang Huang and Egor Kraev"

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(FloatTensor)

def get_attn_padding_mask(seq_q, seq_k, num_actions=Constants.PAD):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(num_actions).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask


class SelfAttentionDecoderStep(nn.Module):
    ''' One continuous step of the decoder '''
    def __init__(self,
                 num_actions,
                 max_seq_len,
                 n_layers=6,#6
                 n_head=6,#8,
                 d_k=16,#64,
                 d_v=16,#64,
                 d_model=128,#512,
                 d_inner_hid=256,#1024,
                 drop_rate=0.1,
                 enc_output_size=76,
                 batch_size=None):

        super().__init__()
        n_position = max_seq_len + 1 # Why the +1? Because of the dummy prev action for first step
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.enc_output_size = enc_output_size
        self.batch_size = batch_size

        self.position_enc = nn.Embedding(n_position, d_model, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_model)
        self.position_enc.weight.requires_grad = False # will this suffice to make them not trainable?


        # TODO: do we want relu after embedding? Probably not; make consistent
        self.embedder = nn.Embedding(
            num_actions, d_model, padding_idx=num_actions-1) # Assume the padding index is the max possible?
        self.dropout = nn.Dropout(drop_rate)

        self.layer_stack = nn.ModuleList([
            DecoderLayerStep(d_model, d_inner_hid, n_head, d_k, d_v, dropout=drop_rate)
            for _ in range(n_layers)])
        # make sure encoder output has correct dim
        if enc_output_size != self.d_model:
            self.enc_output_transform = to_gpu(nn.Linear(enc_output_size, self.d_model))
        else:
            self.enc_output_transform = lambda x: x
        self.dec_output_transform = to_gpu(nn.Linear(self.d_model, num_actions))
        self.all_actions = None
        self.output_shape = [None, self.max_seq_len, num_actions]

    def encode(self, last_action, last_action_pos):
        '''
        Encode the input (last action) into a vector
        :param last_action: batch x 1, batch of sequences of length 1 of ints
        :param last_action_pos: int
        :return: FloatTensor batch x num_actions
        '''
        if last_action_pos > 0:  # if we're not at step 0
            # Word embedding look up
            dec_input = self.embedder(last_action)
            # Position Encoding addition
            batch_size = last_action.size()[0]
            pos_enc = self.position_enc( torch.ones_like(last_action) * last_action_pos# torch.from_numpy(np.array([last_action_pos]))
                                            #(torch.ones(1,1)*last_action_pos).type(LongTensor)
                                        ).expand(batch_size,1,self.d_model)
            dec_input += pos_enc
        else: # just return
            dec_input = torch.zeros_like(self.embedder(torch.zeros_like(last_action)))

        return dec_input

    def forward(self, last_action,
                #last_action_pos=None,
                src_seq=None,
                return_attns=False,
                remember_step=True):
        '''
        Does one continuous step of the decoder, waiting for a policy to then pick an action from
        its output and call it again
        :param last_action: batch of ints: last action taken
        :param last_action_pos: int: num of steps since last reset, is 0 when this is the first action!
        :param src_seq: if enc_output is 2-dim, ignored; else used to check for padding, to make padding mask
        :param return_attns:
        :return:
        '''
        # control that we don't exceed
        if self.n == self.max_seq_len:
            raise StopIteration()


        if self.n > 0:
            last_action = (last_action.unsqueeze(1)).type(LongTensor)
        else: # very first call, last action is meaningless
            last_action = ((torch.ones(len(last_action), 1)) * -1).type(LongTensor)

        #last_action = (last_action.unsqueeze(1)).type(LongTensor)

        if self.all_actions is None:
            new_all_actions = last_action
        else:
            new_all_actions = torch.cat([self.all_actions,last_action], dim=1)

        dec_slf_attn_pad_mask = get_attn_padding_mask(last_action, new_all_actions)

        #TODO: double-check, is this legit?
        dec_input = self.encode(last_action, self.n)# last_action_pos)

        if return_attns:
            dec_slf_attns, dec_enc_attns = [], []

        # TODO: treat new input as a batch of vectors throughout!
        dec_output = dec_input[:,0,:]
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output,
                                                               slf_attn_mask=dec_slf_attn_pad_mask,
                                                               dec_enc_attn_mask=self.dec_enc_attn_pad_mask,
                                                               remember_step=remember_step)

            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]
        if remember_step:
            self.n += 1
            self.all_actions = new_all_actions

        # As the output 'sequence' only contains one step, get rid of that dimension
        if return_attns:
            return self.dec_output_transform(dec_output), dec_slf_attns, dec_enc_attns
        else:
            return self.dec_output_transform(dec_output)

    def init_encoder_output(self, z):
        #self.z_size = z.size()[-1]
        self.n = 0
        if z is not None:
            self.enc_output = self.enc_output_transform(z)

            if len(self.enc_output.shape) == 2:
                # make encoded vector look like a sequence
                # this is the case we support at the moment, encoder output is just a vector
                self.enc_output = torch.unsqueeze(self.enc_output, 1)
                # TODO: check that mask convention is 1 = mask, 0=leave
                # as each enc_input sequence has length 1, don't need to mask
                self.dec_enc_attn_pad_mask = torch.zeros(self.enc_output.size()[0], 1, 1).type(ByteTensor)
            else:
                raise NotImplementedError()
                # dec_enc_attn_pad_mask = get_attn_padding_mask(last_action, src_seq)
        else:
            self.enc_output = z
            self.dec_enc_attn_pad_mask = None

        self.all_actions = None
        for m in self.layer_stack:
            m.init_encoder_output(self.enc_output)

