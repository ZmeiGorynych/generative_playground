import gzip, pickle


class MoleculeSaver:
    def __init__(self, filename, gzip=True):
        self.data = []
        self.filename = filename
        self.gzip = gzip

    def __call__(self, inputs, model, outputs, loss_fn, loss):
        rewards = outputs['rewards'].detach().cpu().numpy()
        smiles = outputs['info'][0]
        self.data += [x for x in zip(smiles, rewards)]
        if self.gzip:
            with gzip.open(self.filename,'wb') as f:
                pickle.dump(self.data, f)
        else:
            with open(self.filename,'wb') as f:
                pickle.dump(self.data, f)
