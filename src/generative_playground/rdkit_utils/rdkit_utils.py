import h5py
import networkx as nx
import numpy as np
from rdkit.Chem import Descriptors, rdmolops
from rdkit.Chem.rdmolfiles import MolFromSmiles
import rdkit.Chem.rdMolDescriptors as desc
from generative_playground.rdkit_utils import sascorer as sascorer
from generative_playground.models.model_settings import get_data_location, get_settings


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

def num_aromatic_rings(smiles):
    '''
    Returns number of atoms in each molecule if valid, None otherwise
    :param smiles: list of strings
    :return: list of float or None, same length
    '''
    mols = mol_from_smiles(smiles)
    sizes = [None if m is None else Descriptors.NumAromaticRings(m) for m in mols]
    return sizes

def num_aliphatic_rings(smiles):
    '''
    Returns number of atoms in each molecule if valid, None otherwise
    :param smiles: list of strings
    :return: list of float or None, same length
    '''
    mols = mol_from_smiles(smiles)
    sizes = [None if m is None else Descriptors.NumAliphaticRings(m) for m in mols]
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
    def __init__(self, invalid_value=-3*3.5, sa_mult=0.0, sa_thresh = 0.5, normalize_scores=False):
        settings = get_settings(True, 'new')
        h5f = h5py.File(settings['data_path'], 'r')
        self.means = np.array(h5f['score_mean'])[:3]
        self.stds = np.array(h5f['score_std'])[:3]
        self.invalid_value = invalid_value
        self.sa_mult = sa_mult
        self.sa_thresh = sa_thresh
        self.normalize_scores = normalize_scores
        h5f.close()

    def get_scores(self, smiles):
        '''
        get normalized score
        :param smiles: a list of strings
        :return: array of normalized scores
        '''
        mols = [MolFromSmiles(s) for s in smiles]
        [print("invalid", s) for (s, mol) in zip(mols, smiles) if mol is None]
        if mols[0] is None:
            print(smiles[0])
        scores = np.array([get_score_components_from_mol(mol) if mol is not None else [float('nan')]*3
                                for mol in mols])
        norm_scores = (scores - self.means) / self.stds
        return scores, norm_scores

    def __call__(self,smiles):
        scores, norm_scores = self.get_scores(smiles)
        if not self.normalize_scores:
            norm_scores = scores
        # extra penalty for low sa_score
        #norm_scores[:,1]*= -1
        norm_score = norm_scores.sum(1) + self.sa_mult * np.array([min(0, x - self.sa_thresh) for x in norm_scores[:,1]])# * sigmoid(2 * (self.sa_thresh - norm_scores[:, 1]))
        for i in range(len(norm_score)):
            if np.isnan(norm_score[i]):
                norm_score[i] = self.invalid_value
        norm_score = np.array([max(x, self.invalid_value + 1) if x!=self.invalid_value else self.invalid_value for x in norm_score])
        return norm_score

def sigmoid(x):
    return 1/(1+np.exp(-x))


def property_scorer(smiles):
    mols = mol_from_smiles(smiles)
    function_list = [
        desc.CalcNumAliphaticRings,
        desc.CalcNumAromaticRings,
        desc.CalcNumRotatableBonds
    ]
    out = []
    for mol in mols:
        if mol is None:
            out.append([0] * (1 + len(function_list)))
        else:
            out.append([1] + list(get_score_components_from_mol(mol)))
                #[f(mol) for f in function_list])

    return np.array(out)