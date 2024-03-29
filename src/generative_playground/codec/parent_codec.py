from generative_playground.utils.gpu_utils import device
import torch
import numpy as np


class GenericCodec:
    def __init__(self):
        # self.vae = None
        # self.decoder = None
        # these have to be supplied by the implementations
        self.grammar = None
        self.MAX_LEN = None
        self.PAD_INDEX = 'max' # TODO: largest possible index is pad index - want to change that to 0 eventually

    # # TODO: model shouldn't be entangled with the codec, take that out!
    # def set_model(self, model):
    #     if model is not None:
    #         self.vae = model
    #         self.vae.eval()
    #         self.decoder = self.vae.decoder

    def is_padding(self, x):
        if self.PAD_INDEX == 'max':
            return x == self.feature_len() -1
        else:
            return x == self.PAD_INDEX

    def feature_len(self):
        return NotImplementedError()

    def decode_from_actions(self, smiles):
        raise NotImplementedError()

    def actions_to_strings(self, actions):
        return self.decode_from_actions(actions)

    def strings_to_actions(self, smiles):
        raise NotImplementedError()

    def encode(self, smiles):
        '''
        Converts a batch of input strings into a batch of latent space vectors
        :param smiles:
        :return:
        '''
        one_hot = self.string_to_one_hot(smiles)
        z_mean = self.vae.encoder.encode(one_hot)
        if type(z_mean) == tuple:
            z_mean = z_mean[0]
        return z_mean

    def decode(self, z):
        '''
        Converts a batch of latent space vectors into a batch of action ints
        :param z: batch x z_size
        :return: smiles: list(str) of len batch, actions: LongTensor batch_size x max_seq_len
        '''
        actions, logits = self.decoder(z.to(device=device))
        smiles = self.decode_from_actions(actions)
        return smiles, actions

    def decode_with_validation(self, z, max_attempts = 10):
        import rdkit
        if type(z) == np.ndarray:
            numpy_output = True
            z = torch.tensor(z)
        else:
            numpy_output=False

        out = []
        actions = []
        for this_z in z:
            for _ in range(max_attempts):
                smiles, action = self.decode(torch.unsqueeze(this_z,0))
                result = rdkit.Chem.MolFromSmiles(smiles[0])
                if result is not None:
                    break
            out.append(smiles[0])
            actions.append(action)
        actions = torch.cat(actions, axis=0)
        if numpy_output:
            actions = actions.cpu().numpy()
        return out, actions

    def action_seq_length(self,these_actions):
        if 'numpy' in str(type(these_actions)):
            # TODO: put a numpy-specific version here, not needing pytorch
            these_actions = these_actions.to(device=device, dtype=torch.long)
        out = torch.zeros(len(these_actions)).to(device=device, dtype=torch.long)#LongTensor((len(these_actions)))
        for i in range(len(these_actions)):
            if these_actions[i][-1] == self._n_chars -1:
                out[i] = torch.nonzero(these_actions[i] == (self._n_chars -1))[0]
            else:
                out[i] = len(these_actions[i])
        return out

    def actions_to_one_hot(self, actions):
        '''

        :param actions: batch_size x max_seq_len np.array(int)
        :return:
        '''
        one_hot = np.zeros((len(actions), self.MAX_LEN, self._n_chars), dtype=np.float32)
        for i in range(len(actions)):
            num_productions = len(actions[i])
            one_hot[i][np.arange(num_productions), actions[i]] = 1.
        return one_hot

    # TODO: purge all one-hot encoding, work with sequences of indices instead
    def string_to_one_hot(self, smiles):
        return self.actions_to_one_hot(self.strings_to_actions(smiles))

