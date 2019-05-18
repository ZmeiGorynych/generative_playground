import inspect
import os

molecules_root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/'


def get_data_location(molecules=True, source='ZINC'):
    if molecules:
        if source == 'ZINC':
            return {'source_data': molecules_root_location + 'data/250k_rndm_zinc_drugs_clean.smi'}
        elif source =='ChEMBL:train':
            return {'source_data': molecules_root_location + 'data/guacamol_v1_train.smiles'}
    else:
        return {'source_data': molecules_root_location + 'data/equation2_15_dataset.txt'}