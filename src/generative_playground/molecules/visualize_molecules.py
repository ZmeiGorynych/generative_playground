
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as desc
from rdkit.Chem.Draw import MolToFile
import os, inspect
from generative_playground.molecules.rdkit_utils.rdkit_utils import NormalizedScorer
import numpy as np

scorer = NormalizedScorer()
root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def model_process_fun(model_out, visdom, n):
    # TODO: rephrase this to return a dict, instead of calling visdom directly
    from rdkit import Chem
    from rdkit.Chem.Draw import MolToFile
    # actions, logits, rewards, terminals, info = model_out
    smiles, valid = model_out['info']
    valid = to_numpy(valid)
    total_rewards = to_numpy(model_out['rewards'])
    if len(total_rewards.shape) > 1:
        total_rewards = total_rewards.sum(1)
    best_ind = np.argmax(total_rewards)
    this_smile = smiles[best_ind]
    mol = Chem.MolFromSmiles(this_smile)
    pic_save_path = os.path.realpath(root_location + '/images/' + 'tmp.svg')
    if mol is not None:
        try:
            MolToFile(mol, pic_save_path, imageType='svg')
            with open(pic_save_path, 'r') as myfile:
                data = myfile.read()
            data = data.replace('svg:', '')
            visdom.append('best molecule of batch', 'svg', svgstr=data)
        except Exception as e:
            print(e)
        scores, norm_scores = scorer.get_scores([this_smile])
        visdom.append('score component',
                      'line',
                      X=np.array([n]),
                      Y=np.array([ [x for x in norm_scores[0]] + [norm_scores[0].sum()] + [scores[0].sum()] + [
                          desc.CalcNumAromaticRings(mol)] ]),
                      opts={'legend': ['logP', 'SA', 'cycle', 'norm_reward', 'reward', 'Aromatic rings']})
        visdom.append('reward',
                      'line',
                      X=np.array([n]),
                      Y=np.array([total_rewards[best_ind]]))
        visdom.append('fraction valid',
                      'line',
                      X=np.array([n]),
                      Y=np.array([valid.mean()]))
        visdom.append('num atoms', 'line',
                      X=np.array([n]),
                      Y=np.array([len(mol.GetAtoms())]))

def to_numpy(x):
    if hasattr(x,'device'): # must be pytorch
        return x.cpu().detach().numpy()
    else:
        return x