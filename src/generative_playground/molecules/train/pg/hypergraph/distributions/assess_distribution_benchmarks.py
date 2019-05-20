from guacamol.assess_distribution_learning import assess_distribution_learning
from generative_playground.molecules.guacamol_utils import DummyMoleculeGenerator
from generative_playground.molecules.lean_settings import get_data_location


train_mols = get_data_location(source="ChEMBL:train")['source_data']
my_gen = DummyMoleculeGenerator('distribution_naive_smiles.zip', maximize_reward=False)
assess_distribution_learning(my_gen, train_mols)