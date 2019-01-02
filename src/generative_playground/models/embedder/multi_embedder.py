import torch.nn as nn
import torch

class MultiEmbedder(nn.Module):
    '''
    Multi-language embedder, uses the first index in each sequence to determine language
    '''
    def __init__(self,
                 num_lang,
                 input_dim,
                 output_dim,
                 index_offset=1,
                 padding_idx = 0):
        super().__init__()
        self.output_dim = output_dim
        self.language_embed = nn.Embedding(num_lang,output_dim) # no padding index needed
        self.index_offset = index_offset
        self.embedders = nn.ModuleList([nn.Embedding(input_dim,
                                                     output_dim,
                                                     padding_idx=padding_idx)
                                        for _ in range(num_lang)])

    def forward(self, x):
        '''
        Embed the sequences, using the first index to pick the embedder
        The output's first element embeds the language
        :param x: batch_size x num_steps long
        :return: batch_size x num_steps x output_dim float
        '''
        languages = x[:,0] - int(self.index_offset)
        embed_languages = self.language_embed(languages.unsqueeze(1))
        embed = torch.zeros(x.size()[0], x.size()[1]-1, self.output_dim,
                            dtype=torch.float32, device=x.device)
        for i, embedder in enumerate(self.embedders):
            embed[languages == i, :, :] = embedder(x[:,1:][languages == i, :])

        out = torch.cat([embed_languages, embed], dim=1)
        return out




