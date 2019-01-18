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
from generative_playground.molecules.train.vae.main_train_vae import train_vae
from generative_playground.molecules.model_settings import get_settings

molecules = True
grammar = True
settings = get_settings(molecules,grammar)

save_file =settings['filename_stub'] + 'dr0.2_attn__.h5'

model, fitter, main_dataset = train_vae(molecules=molecules,
                                        BATCH_SIZE=10,  # it's a bit of a GPU RAM hog
                                        drop_rate=0.2,
                                        save_file=save_file,
                                        sample_z=False,
                                        reg_weight=1, # with 0.01 and do 0.1 had real trouble generalizing?
                                        encoder_type='attention',
                                        decoder_type='attention',
                                        lr=1e-4,
                                        plot_prefix='attn do=0.2 no_sam 1e-4',
                                        preload_weights=False)

while True:
    next(fitter)

