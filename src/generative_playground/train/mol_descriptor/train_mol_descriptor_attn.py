#!/home/carnd/anaconda3/envs/torch/bin/python

# One upside for calling this as shell script rather than as 'python x.py' is that
# you can see the script name in top/ps - useful when you have a bunch of python processes

try:
    import generative_playground
except:
    import sys, os, inspect
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../../..')
    sys.path.append('../../../../../transformer_pytorch')

from generative_playground.train.mol_descriptor.main_train_mol_descriptor import train_mol_descriptor
from generative_playground.models.model_settings import get_settings

molecules = True
grammar = True
settings = get_settings(molecules,grammar)

save_file =settings['filename_stub'] + 'attn_mol_desc.h5'
model, fitter, _ = train_mol_descriptor(grammar = True,
              EPOCHS = 100,
              BATCH_SIZE = 40,
              lr = 5e-5,
                                        gradient_clip=1,
              drop_rate = 0.2,
              plot_ignore_initial = 0,
              save_file = save_file,
              preload_file = None,
              encoder_type='attention',
              plot_prefix = 'attention ',
              dashboard = 'mol_desc',
              preload_weights=False)

while True:
    next(fitter)

