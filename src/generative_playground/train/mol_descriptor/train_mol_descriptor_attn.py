#!/home/carnd/anaconda3/envs/torch/bin/python

# One upside for calling this as shell script rather than as 'python x.py' is that
# you can see the script name in top/ps - useful when you have a bunch of python processes
import sys, os, inspect
try:
    import generative_playground
except:
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../../..')
    sys.path.append('../../../../../transformer_pytorch')

from generative_playground.train.mol_descriptor.main_train_mol_descriptor import train_mol_descriptor
from generative_playground.models.model_settings import get_settings
from generative_playground.data_utils.data_sources import MultiDatasetFromHDF5, IncrementingHDF5Dataset

molecules = True
grammar = True
settings = get_settings(molecules,grammar)

save_file =settings['filename_stub'] + 'attn_mol_desc.h5'

root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
root_location = root_location + '/../'
save_path = root_location + 'pretrained/paper/policy_gradient_baseline.h5'
aux_dataset = IncrementingHDF5Dataset(save_path)

model, fitter, _ = train_mol_descriptor(grammar=True,
                                        EPOCHS=100,
                                        BATCH_SIZE=10,
                                        lr=5e-5,
                                        gradient_clip=0.1,
              drop_rate = 0.1,
              plot_ignore_initial = 0,
              save_file = save_file,
              preload_file = None,
              encoder_type='attention',
              plot_prefix = 'attention ',
              dashboard = 'mol_desc',
              preload_weights=False,
                                        aux_dataset=None)#aux_dataset)

while True:
    next(fitter)

