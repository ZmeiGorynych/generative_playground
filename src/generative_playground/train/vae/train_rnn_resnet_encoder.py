#!/home/carnd/anaconda3/envs/torch/bin/python

# One upside for calling this as shell script rather than as 'python x.py' is that
# you can see the script name in top/ps - useful when you have a bunch of python processes

try:
    import generative_playground
except:
    import sys, os, inspect
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../..')
    sys.path.append('../../generative_playground')
    sys.path.append('../../transformer_pytorch')
from generative_playground.train.vae.main_train_vae import train_vae

from generative_playground.models.model_settings import get_settings

molecules = True
grammar = True
settings = get_settings(molecules,grammar)

save_file =settings['filename_stub'] + 'dr0.4_rnn_resnet.h5'

model, fitter, main_dataset = train_vae(molecules=molecules,
                          BATCH_SIZE=200,
                          drop_rate=0.4,
                          save_file=save_file,
                          sample_z=False,
                          rnn_encoder='rnn',
                        decoder_type='action_resnet',
                          lr=1e-4,
                          plot_prefix='rnn_resnet do=0.4 1e-4')

while True:
    next(fitter)

