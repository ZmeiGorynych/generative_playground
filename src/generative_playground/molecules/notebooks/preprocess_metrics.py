import pickle
import h5py
import numpy as np
import sys
sys.path.append('../..')
sys.path.append('../../../../transformer_pytorch')
from generative_playground.molecules.rdkit_utils.rdkit_utils  import NormalizedScorer
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem import Descriptors

from collections import OrderedDict
runs = OrderedDict(reversed([('Unconstrained',{'file':'pg_smiles_no_anchor.h5','range': (200, 300)}),
                   ('Weak Anchor, SA penalty',{'file':'pg_smiles_anchor_wweak_sa20.h5','range':(1600, 1700)}),
                   ('Weak Anchor, SA and aromatic cycle penalty',{'file':'pg_smiles_anchor_wweak_sa20_cycle.h5','range':(3850, 3950)}),
                   ('Strong Anchor, SA penalty',{'file':'pg_smiles_anchor_weak_sa20.h5','range':(1900, -1)})]))

def get_smiles_bunches(runs):
    bunches = []
    for key, value in runs.items():
        smiles_file = h5py.File('../train/pretrained/paper/' + runs[key]['file'],'r')
        smiles = np.array(smiles_file['smiles'])[runs[key]['range'][0]*40:runs[key]['range'][1]*40]
        bunches.append((key,smiles))
    return bunches

smiles_bunches = get_smiles_bunches(runs)
kusner1 ='CCCc1ccc(I)cc1C1CCC-c1'
kusner2 ='CC(C)CCCCCc1ccc(Cl)nc1'
kusner3 ='CCCc1ccc(Cl)cc1CCCCOC'
jin = 'c1c(Cl)ccc2c1cc(C(=O)C)c(C(=O)Nc3cc(NC(=O)Nc4cc(c5ccccc5(C))ccc4)ccc3)c2'

smiles_bunches = [('Kusner et al.', [kusner1, kusner2, kusner3]),
                  ('Jin et al.',[jin])] \
                    + smiles_bunches
#smiles_bunches

import copy
def get_all_metrics(smiles):
    mols = [MolFromSmiles(s) for s in smiles]
    scorer = NormalizedScorer()
    scores, norm_scores = scorer.get_scores_from_mols(mols)
    arom_rings = np.array([Descriptors.NumAromaticRings(m) for m in mols])
    metrics = np.concatenate([scores.sum(axis=1)[:, None],
                              norm_scores.sum(axis=1)[:, None],
                              scores[:, 1][:, None],
                              norm_scores[:, 1][:, None],
                              arom_rings[:, None]],
                             axis=1)
    return (smiles, metrics)


def get_best(metrics_ext, name, num_best=1):
    smiles, metrics = metrics_ext
    labels = ['1st', '2nd', '3rd', '4th', '5th']
    # print(num_best)
    metric_rows = []
    neg_sa = metrics[:, 3] < 0  # we're only interested in positive SA
    pos_sa_scores = copy.deepcopy(metrics[:, 1])
    pos_sa_scores[neg_sa] = -100
    for i in range(num_best):
        # print(i)
        best_ind = np.argmax(pos_sa_scores)
        if num_best > 1:
            this_name = name + ' ' + labels[i]
        else:
            this_name = name
        metric_rows.append((this_name, metrics[best_ind, :], smiles[best_ind]))
        pos_sa_scores[best_ind] = -100
    return metric_rows

def generate_metrics_list(smiles_bunches):
    metrics_list = []
    for name, s in smiles_bunches:
        file_name = name.replace(' ','').replace('.','').replace(',','') + '.pickle'
        print(file_name)
        try:
            with open(file_name,'rb') as f:
                metrics_ext = pickle.load(f)
        except:
            print('reading from file...')
            metrics_ext = get_all_metrics(s)
            with open(file_name, 'wb') as f:
                pickle.dump(metrics_ext, f)

        metrics_list += get_best(metrics_ext, name, num_best = 3 if 'cycle' in name or 'Strong' in name else 1)
        print(name, metrics_list[-1])
    return metrics_list

metrics_list = generate_metrics_list(smiles_bunches)
with open('metrics_list.pickle', 'wb') as f:
    pickle.dump(metrics_list, f)

print(metrics_list)