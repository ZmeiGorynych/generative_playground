import torch
from torch import nn as nn
import numpy as np


from generative_playground.models.transformer import Constants as Constants
from generative_playground.utils.gpu_utils import to_gpu, LongTensor


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class InputSequenceNormalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src_seq):
        '''
        Converts sequence from 3d one-hot to 2d longs, if necessary; and generates the src_seq_for_masking for the transformer
        :param src_seq: batch x num_steps x feature_dim long, or batch x num_steps long or float
        :return:
        '''
        if src_seq.dtype == torch.int64:  # indices of discrete actions
            if len(src_seq.size()) == 3:  # if got a one-hot encoded vector
                src_seq = torch.max(src_seq, 2)[-1]  # argmax
            src_seq_for_masking = src_seq
        elif src_seq.dtype == torch.float32 and len(src_seq.size()) == 3:  # if the input is continuous
            src_seq_for_masking = torch.ones(src_seq.size()[:2], device=src_seq.device)

        return src_seq, src_seq_for_masking


class Embedder(nn.Module):
    def __init__(self,
                 n_max_seq,
                 n_src_vocab,  # feature_len
                 d_model=512,  # 128,#
                 padding_idx=Constants.PAD,  # TODO: remember to set this to n_src_vocab-1 when calling from my code!
                 encode_position=True,
                 include_learned=True,
                 include_predefined=False,
                 float_input = False,
                 predefined_dim=256, # polyglot embeddings
                 custom_embedder = None
                 ):
        super().__init__()
        n_position = n_max_seq + 1
        self.include_learned = include_learned # ignored for now
        self.include_predefined = include_predefined
        self.encode_position = encode_position
        self.normalizer = InputSequenceNormalizer()
        self.position_enc = nn.Embedding(n_position, d_model, padding_idx=padding_idx)
        self.position_enc.weight.data = position_encoding_init(n_position, d_model)
        if custom_embedder is None:
            self.src_word_emb = nn.Embedding(n_src_vocab, d_model, padding_idx=padding_idx)
        else:
            self.src_word_emb = custom_embedder

        if float_input:
            if self.include_predefined:
                start_dim = predefined_dim + n_src_vocab
            else:
                start_dim = n_src_vocab
        else:
            if self.include_predefined:
                start_dim = predefined_dim + d_model
            else:
                start_dim = d_model # never used

        self.transform_to_d_model = nn.Linear(start_dim, d_model)

    def forward(self, src_seq, src_pos=None):
        '''
        Embed the source sequence, with optional position specification
        :param src_seq: batch x num_steps long or batch x num_steps x src_vocab one-hot or float
        :param src_pos: batch x num_steps long, or None
        :return:
        '''
        if isinstance(src_seq, tuple):
            predefined_emb = src_seq[1]
            src_seq = src_seq[0]

        src_seq, src_seq_for_masking = self.normalizer(src_seq)
        if src_seq.dtype == torch.int64:  # indices of discrete actions
            enc_input = self.src_word_emb(src_seq)
            if self.include_predefined:
                enc_input = torch.cat([enc_input, predefined_emb], dim=2)
                enc_input = self.transform_to_d_model(enc_input)
        elif src_seq.dtype == torch.float32 and len(src_seq.size()) == 3:  # if the input is continuous
            enc_input = src_seq
            if self.include_predefined:
                enc_input = torch.cat([enc_input, predefined_emb], dim=2)
            enc_input = self.transform_to_d_model(enc_input)

        # Position Encoding addition
        if self.encode_position:
            if src_pos is None:
                batch_size = src_seq.size()[0]
                seq_len = src_seq.size()[1]
                src_pos = to_gpu(torch.arange(seq_len).unsqueeze(0).expand(batch_size,seq_len).type(LongTensor))

            enc_input += self.position_enc(src_pos)
        return enc_input, src_seq_for_masking