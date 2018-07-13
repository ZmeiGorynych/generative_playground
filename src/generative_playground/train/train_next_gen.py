#!/home/carnd/anaconda3/envs/torch/bin/python

# One upside for calling this as shell script rather than as 'python x.py' is that
# you can see the script name in top/ps - useful when you have a bunch of python processes

try:
    import generative_playground
except:
    import sys, os, inspect
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../..')
from generative_playground.data_utils.data_sources import IncrementingHDF5Dataset
from generative_playground.visdom_helper.visdom_helper import Dashboard
from generative_playground.train.vae.main_train_vae import train_vae
from generative_playground.train.main_train_validity import train_validity
from generative_playground.models.model_settings import get_settings
from generative_playground.codec.grammar_codec import ZincGrammarModel
from generative_playground.rdkit_utils.rdkit_utils import fraction_valid
from generative_playground.models.simple_models import DenseHead
import numpy as np
from generative_playground.gpu_utils import to_gpu

molecules = True
grammar = True
settings = get_settings(molecules, grammar)

dash_name = 'test'
visdom = Dashboard(dash_name)
model, fitter, main_dataset = train_vae(molecules=True,
                          grammar=True,
                          BATCH_SIZE=150,
                          drop_rate=0.3,
                          sample_z=True,
                          save_file='next_gen.h5',
                          rnn_encoder=False,
                          lr=5e-4,
                          plot_prefix='RNN enc lr 1e-4',
                          dashboard=dash_name,
                          preload_weights=False)
# this is a wrapper for encoding/decodng
grammar_model = ZincGrammarModel(model=model)
validity_model = to_gpu(DenseHead(model.encoder,
                                  body_out_dim=settings['z_size'],
                                  drop_rate=0.3))

valid_smile_ds = IncrementingHDF5Dataset('valid_smiles.h5', valid_frac=0.1)
invalid_smile_ds = IncrementingHDF5Dataset('invalid_smiles.h5', valid_frac=0.1)

valid_fitter = train_validity(grammar = grammar,
              model = validity_model,
              EPOCHS = 100,
              BATCH_SIZE = 40,
              lr = 1e-4,
              main_dataset = main_dataset,
              new_datasets = (valid_smile_ds, invalid_smile_ds),
              save_file = None,
              plot_prefix = 'valid_model',
              dashboard = dash_name,
              preload_weights=False)

# TODO: collect the smiles strings too, just for kicks, into hdf5!
# f_valid = open("valid_1.smi", "a+")
# f_invalid = open("invalid_1.smi",'a+')
count = 0
sm_metrics = None
while True:
    # this does one train step
    next(fitter)
    mock_latent_points = np.random.normal(size=(100,settings['z_size']))
    mock_smiles, mock_one_hot = grammar_model.decode_ext(mock_latent_points, validate=False, max_attempts=1)
    if len([s for s in mock_smiles if s == '']):
        raise ValueError("With the new masking, sequences should always complete!")

    metrics, mols = fraction_valid(mock_smiles) # frac_valid, avg_len, max_len
    is_valid = [mol is not None for mol in mols]
    if sm_metrics is None:
        sm_metrics = metrics
    else:
        sm_metrics = [0.9*sm + 0.1*m for sm,m in zip(sm_metrics,metrics)]
    visdom.append('molecule validity',
               'line',
               X=np.array([count]),
               Y=np.array([sm_metrics]),
               opts={'legend': ['num_valid','avg_len','max_len']})
    #valid_smiles = [s for s, valid in zip(mock_smiles,is_valid) if valid]
    valid_smile_ds.append(mock_one_hot[np.array(is_valid)])
    #invalid_smiles = [s for s in mock_smiles if s not in valid_smiles]
    invalid_smile_ds.append(mock_one_hot[np.array(is_valid)==False])

    next(valid_fitter)
    count +=1


        # for s in valid:
        #     f_valid.write(s+"\n")
        #
        # for s in invalid:
        #     f_invalid.write(s+"\n")

