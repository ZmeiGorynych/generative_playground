import numpy as np

try:
    import generative_playground
except:
    import os, inspect, sys
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../..')

from generative_playground.models.model_settings import get_settings, get_model
from generative_playground.data_utils.data_sources import IncrementingHDF5Dataset
from generative_playground.rdkit_utils.rdkit_utils import get_score_components

# change this to true to produce the equation dataset
molecules = True
# change this to True to get string-based encodings instead of grammar-based
grammar = True

# can't define model class inside settings as it itself uses settings a lot
_, my_model = get_model(molecules,grammar)
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
for i in range(0, len(L), step):#for i in range(0, 1000, 2000):
    print('Processing: i=[' + str(i) + ':' + str(i + step) + ']')
    these_smiles = L[i:min(i + step,len(L))]
    these_actions = my_model.string_to_actions(these_smiles)
    action_seq_length = my_model.action_seq_length(these_actions)
    onehot = my_model.actions_to_one_hot(these_actions)
    append_data = {'smiles': np.array(these_smiles, dtype='S'),
                   'actions': these_actions,
                   'valid': np.ones((len(these_smiles))),
                   'seq_len': action_seq_length,
                   'data': onehot}
    if molecules:
        raw_scores = np.array([get_score_components(s) for s in these_smiles])
        append_data['raw_scores'] = raw_scores

    ds.append(append_data)

if molecules:
    # also calculate mean and std of the scores, to use in the ultimate objective
    raw_scores = np.array(ds.h5f['raw_scores'])
    score_std = raw_scores.std(0)
    score_mean = raw_scores.mean(0)
    ds.append_to_dataset('score_std',score_std)
    ds.append_to_dataset('score_mean', score_mean)

print('success!')

