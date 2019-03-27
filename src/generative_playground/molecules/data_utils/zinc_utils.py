from generative_playground.molecules.lean_settings import get_data_location
from rdkit.Chem import MolFromSmiles, AddHs, MolToSmiles, RemoveHs, Kekulize, BondType

def get_zinc_smiles(num=10):
    L = []
    settings = get_data_location(molecules=True)
    # Read in the strings
    with open(settings['source_data'], 'r') as f:
        for line in f:
            line = line.strip()
            L.append(line)
            if len(L) >= num:
                break

    print('loaded data!')
    return L

def get_zinc_molecules(num=10):
    smiles = get_zinc_smiles(num)
    return [MolFromSmiles(s) for s in smiles]