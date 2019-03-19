import numpy as np
# from generative_playground.models.decoders import OneStepDecoderContinuous, SimpleDiscreteDecoder
# from generative_playground.models.policy import SoftmaxRandomSamplePolicy
from generative_playground.codec.parent_codec import GenericCodec

class CharacterCodec(GenericCodec):
    def __init__(self,
                 model = None,
                 max_len = None,
                 charlist = None
                 ):
        self.set_model(model)
        self.charlist = charlist
        self._n_chars = len(charlist)
        # below is the shared code
        self.MAX_LEN = max_len
        self._char_index = {}
        for ix, char in enumerate(self.charlist):
            self._char_index[char] = ix

    def feature_len(self):
        return len(self.charlist)

    def strings_to_actions(self, smiles):
        actions = [[self._char_index[c] for c in entry] for entry in smiles]
        # now pad them to full length
        actions = np.array([a + [self._n_chars - 1] * (self.MAX_LEN - len(a)) for a in actions]).astype(int)
        return actions
    # def actions_to_one_hot(self, actions):
    #     """ Encode a list of smiles strings into the latent space """
    #     #indices = [np.array([self._char_index[c] for c in entry], dtype=int) for entry in smiles]
    #     one_hot = np.zeros((len(actions), self.MAX_LEN, len(self.charlist)), dtype=np.float32)
    #     for i in range(len(indices)):
    #         num_productions = len(indices[i])
    #         one_hot[i][np.arange(num_productions),indices[i]] = 1.
    #         one_hot[i][np.arange(num_productions, self.MAX_LEN),-1] = 1.
    #     return one_hot

    def decode_from_actions(self, actions):
        char_matrix = np.array(self.charlist)[np.array(actions, dtype=int)]
        return [''.join(ch).strip() for ch in char_matrix]




