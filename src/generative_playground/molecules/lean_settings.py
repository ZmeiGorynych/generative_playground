import inspect
import os

molecules_root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/'


def get_data_location(molecules=True):
    if molecules:
        return {'source_data': molecules_root_location + 'data/250k_rndm_zinc_drugs_clean.smi'}
    else:
        return {'source_data': molecules_root_location + 'data/equation2_15_dataset.txt'}