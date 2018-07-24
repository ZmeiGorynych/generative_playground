from generative_playground.utils.gpu_utils import FloatTensor, to_gpu, LongTensor
import torch
import numpy as np

class GenericCodec:
    def encode(self, smiles):
        one_hot = self.string_to_one_hot(smiles)
        z_mean = self.vae.encoder.encode(one_hot)
        if type(z_mean) == tuple:
            z_mean = z_mean[0]
        return z_mean

    def decode(self, z):
        '''
        Converts a batch of latent vectors into a batch of action ints
        :param z: batch x z_size
        :return: smiles: list(str) of len batch, actions: LongTensor batch_size x max_seq_len
        '''
        actions, logits = self.decoder(to_gpu(z))
        smiles = self.decode_from_actions(actions)
        return smiles, actions

    def decode_with_validation(self, z, max_attempts = 10):
        import rdkit
        if type(z) == np.ndarray:
            numpy_output = True
            z = FloatTensor(z)
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
            these_actions = LongTensor(these_actions)
        out = LongTensor((len(these_actions)))
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

    # todo: move to supeclass
    def string_to_one_hot(self, smiles):
        return self.actions_to_one_hot(self.string_to_actions(smiles))

