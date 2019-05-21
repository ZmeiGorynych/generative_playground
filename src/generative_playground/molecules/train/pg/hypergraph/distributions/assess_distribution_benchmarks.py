from guacamol.assess_distribution_learning import assess_distribution_learning
from generative_playground.molecules.guacamol_utils import DummyMoleculeGenerator
from generative_playground.molecules.lean_settings import get_data_location


# # naive sampling
# train_mols = get_data_location(source="ChEMBL:train")['source_data']
# my_gen = DummyMoleculeGenerator('distribution_naive_smiles.zip', maximize_reward=False)
# assess_distribution_learning(my_gen, train_mols)

# stable-ish run with discriminator sampling
train_mols = get_data_location(source="ChEMBL:train")['source_data']
my_gen = DummyMoleculeGenerator('distribution_discr_copy.zip',#eps0.2_smiles.zip',
                                maximize_reward=False,
                                keep_last=1e4)
assess_distribution_learning(my_gen, train_mols)