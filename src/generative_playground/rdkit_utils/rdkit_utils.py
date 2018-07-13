import h5py
import networkx as nx
import numpy as np
from rdkit.Chem import Descriptors, rdmolops
from rdkit.Chem.rdmolfiles import MolFromSmiles

from generative_playground.rdkit_utils import sascorer as sascorer


def fraction_valid(smiles):
    mols = mol_from_smiles(smiles)
    valid_lens = [len(m.GetAtoms()) for m in mols if m is not None]
    num_valid = len(valid_lens)
    avg_len = sum(valid_lens) / (len(valid_lens) + 1e-6)
    max_len = 0 if not len(valid_lens) else max(valid_lens)
    return (num_valid, avg_len, max_len), mols

def mol_from_smiles(smiles):
    if type(smiles)=='str':
        return MolFromSmiles(smiles)
    else: # assume we have a list-like
        return [MolFromSmiles(s) for s in smiles]

def num_atoms(smiles):
    '''
    Returns number of atoms in each molecule if valid, None otherwise
    :param smiles: list of strings
    :return: list of float or None, same length
    '''
    mols = mol_from_smiles(smiles)
    sizes = [None if m is None else len(m.GetAtoms()) for m in mols]
    return sizes


def get_score_components_from_mol(this_mol):
    try:
        logP = Descriptors.MolLogP(this_mol)
    except:
        logP = 0.0
    SA_score = -sascorer.calculateScore(this_mol)
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(this_mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length
    return logP, SA_score, cycle_score


def get_score_components(smiles):
    '''
    Get the non-normalized score components
    :param smiles: a VALID smiles string
    :return: a tuple of floats
    '''
    this_mol = MolFromSmiles(smiles)
    return get_score_components_from_mol(this_mol)


class NormalizedScorer:
    def __init__(self, filename, invalid_value=-2):
        h5f = h5py.File(filename, 'r')
        self.means = np.array(h5f['score_mean'])
        self.stds = np.array(h5f['score_std'])
        self.invalid_value = invalid_value
        h5f.close()

    def __call__(self, smiles):
        '''
        get normalized score
        :param smiles: a list of strings
        :return: array of normalized scores
        '''
        mols = [MolFromSmiles(s) for s in smiles]
        scores = np.array([get_score_components_from_mol(mol) if mol is not None else [float('nan')]*3
                                for mol in mols])
        norm_scores = ((scores - self.means)/self.stds).sum(1)
        for i in range(len(norm_scores)):
            if np.isnan(norm_scores[i]):
                norm_scores[i] = self.invalid_value
        return norm_scores