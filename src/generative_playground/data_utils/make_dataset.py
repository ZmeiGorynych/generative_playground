import numpy as np
import h5py

try:
    import generative_playground
    import transformer
except:
    import os, inspect, sys
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../..')
    sys.path.append('../../../../transformer_pytorch')

from generative_playground.models.model_settings import get_settings, get_model
from generative_playground.data_utils.data_sources import IncrementingHDF5Dataset
from generative_playground.rdkit_utils.rdkit_utils import get_score_components_from_mol

# change this to False to produce the equation dataset
molecules = True
# change this to False to get string-based encodings instead of grammar-based
grammar = 'new' #True#True# true for the grammar used by Kusner et al

# can't define model class inside settings as it itself uses settings a lot
_, my_model = get_model(molecules,grammar)
def pre_parser(x):
    try:
        return next(my_model._parser.parse(x))
    except Exception as e:
        return None

settings = get_settings(molecules,grammar)
MAX_LEN = settings['max_seq_length']
feature_len = settings['feature_len']
dest_file = settings['data_path']
source_file = settings['source_data']

# Read in the strings
f = open(source_file,'r')
L = []
for line in f:
    line = line.strip()
    L.append(line)
f.close()

# convert to one-hot and save, in small increments to save RAM
#dest_file = dest_file.replace('.h5','_new.h5')
ds = IncrementingHDF5Dataset(dest_file)

step = 100
dt = h5py.special_dtype(vlen=str)     # PY3 hdf5 datatype for variable-length Unicode strings

for i in range(0, len(L), step):#for i in range(0, 1000, 2000):
    print('Processing: i=[' + str(i) + ':' + str(i + step) + ']')
    these_indices = list(range(i, min(i + step,len(L))))
    these_smiles = L[i:min(i + step,len(L))]
    if grammar=='new': # have to weed out non-parseable strings
        tokens = [my_model._tokenize(s.replace('-c','c')) for s in these_smiles]
        these_smiles, these_indices = list(zip(*[(s,ind) for s,t,ind in zip(these_smiles, tokens, these_indices) if pre_parser(t) is not None]))
        print(len(these_smiles))
    these_actions = my_model.string_to_actions(these_smiles)
    action_seq_length = my_model.action_seq_length(these_actions)
    onehot = my_model.actions_to_one_hot(these_actions)
    append_data = {'smiles': np.array(these_smiles, dtype=dt),
                   'indices': np.array(these_indices),
                   'actions': these_actions,
                   'valid': np.ones((len(these_smiles))),
                   'seq_len': action_seq_length,
                   'data': onehot}
    if molecules:
        from rdkit.Chem.rdmolfiles import MolFromSmiles
        mols = [MolFromSmiles(s) for s in these_smiles]
        raw_scores = np.array([get_score_components_from_mol(m) for m in mols])
        append_data['raw_scores'] = raw_scores
        num_atoms = np.array([len(m.GetAtoms()) for m in mols])
        append_data['num_atoms'] = num_atoms

    ds.append(append_data)

if molecules:
    # also calculate mean and std of the scores, to use in the ultimate objective
    raw_scores = np.array(ds.h5f['raw_scores'])
    score_std = raw_scores.std(0)
    score_mean = raw_scores.mean(0)
    ds.append_to_dataset('score_std',score_std)
    ds.append_to_dataset('score_mean', score_mean)

print('success!')

