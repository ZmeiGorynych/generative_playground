import torch.nn as nn
import torch

class MultiEmbedder(nn.Module):
    '''
    Multi-language embedder, uses the first index in each sequence to determine language
    '''
    def __init__(self,
                 languages,
                 lang_mapping,
                 input_dim,
                 output_dim,
                 index_offset=1,
                 padding_idx=0):
        super().__init__()
        self.output_dim = output_dim
        self.language_map = {key: value - int(index_offset) for key, value in lang_mapping.items()
                             if key in languages}
        self.language_embed = nn.Embedding(len(lang_mapping)-2,output_dim) # no padding index needed
        self.index_offset = index_offset
        self.embedders = nn.ModuleDict({lang: nn.Embedding(input_dim,
                                                     output_dim,
                                                     padding_idx=padding_idx)
                                        for lang in languages})

    def forward(self, x):
        '''
        Embed the sequences, using the first index to pick the embedder
        The output's first element embeds the language
        :param x: batch_size x num_steps long
        :return: batch_size x num_steps x output_dim float
        '''
        language_codes = x[:,0] - int(self.index_offset)
        embed_languages = self.language_embed(language_codes.unsqueeze(1))
        embed = torch.zeros(x.size()[0], x.size()[1]-1, self.output_dim,
                            dtype=torch.float32, device=x.device)
        for lang, embedder in self.embedders.items():
            i = self.language_map[lang]
            #if  in language_codes:
            embed[language_codes == i, :, :] = embedder(x[:,1:][language_codes == i, :])

        out = torch.cat([embed_languages, embed], dim=1)
        return out




