from guacamol.assess_distribution_learning import assess_distribution_learning
from generative_playground.molecules.guacamol_utils import DummyMoleculeGenerator
from generative_playground.molecules.lean_settings import get_data_location

# # naive sampling
# train_mols = get_data_location(source="ChEMBL:train")['source_data']
# my_gen = DummyMoleculeGenerator('distribution_naive_smiles.zip', maximize_reward=False)
# assess_distribution_learning(my_gen, train_mols)

# stable-ish run with discriminator sampling
#
files = [
    ['distribution_naive_no_priors_smiles.zip', 'distribution_naive_no_priors_smiles_2.zip'],
    'distribution_naive_uncond_priors_smiles.zip',  # is the one with unconditional priors
    'distribution_naive_smiles.zip',  # is the one with conditional priors
    'distribution_discr_eps0.2_smiles.zip']

train_mols = get_data_location(source="ChEMBL:train")['source_data']
my_gen = DummyMoleculeGenerator(files[0],  # eps0.2_smiles.zip',
                                maximize_reward=False,
                                keep_last=1e4)
assess_distribution_learning(my_gen, train_mols)
