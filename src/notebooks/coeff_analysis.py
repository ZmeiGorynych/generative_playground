import os, glob, pickle, sys
import torch

if '/home/ubuntu/shared/GitHub' in sys.path:
    sys.path.remove('/home/ubuntu/shared/GitHub')
from generative_playground.models.pg_runner import PolicyGradientRunner
from generative_playground.models.decoder.decoder import get_node_decoder
snapshot_dir = os.path.realpath('../generative_playground/molecules/train/pretrained')
root_name = 'a3genetic8_v2_lr_0.02_ew_0.0'
files = glob.glob(snapshot_dir + '/Ascope8*.h5')#_runner.zip')
coeffs = {}
for file in files:
    print(file)
    model = get_node_decoder('hypergraph:hyper_grammar_guac_10k_with_clique_collapse.pickle',
                     decoder_type='graph_conditional',
                     priors='conditional',
                             batch_size=2)[0]
    model.load_state_dict(torch.load(file))

    coeffs[file] = model.stepper.model.get_params_as_vector()

with open(snapshot_dir + '/Ascope8_coeff_summary.pkl','wb') as f:
    pickle.dump(coeffs, f)