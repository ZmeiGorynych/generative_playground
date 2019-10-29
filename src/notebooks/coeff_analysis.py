import os, glob, pickle, sys
import torch

if '/home/ubuntu/shared/GitHub' in sys.path:
    sys.path.remove('/home/ubuntu/shared/GitHub')
from generative_playground.models.pg_runner import PolicyGradientRunner
from generative_playground.models.decoder.decoder import get_node_decoder
snapshot_dir = os.path.realpath('../generative_playground/molecules/train/genetic/data')
root_name = 'Ascan8_v2_lr0.03_ew0.1'#'AA2scan8_v2_lr0.1_ew0.1' #
files = glob.glob(snapshot_dir + '/' + root_name + '/*_runner.zip')
coeffs = {}
for file in files:
    print(file)
    model = PolicyGradientRunner.load(file)
    coeffs[file] = model.params
    # model = get_node_decoder('hypergraph:hyper_grammar_guac_10k_with_clique_collapse.pickle',
    #                  decoder_type='graph_conditional',
    #                  priors='conditional',
    #                          batch_size=2)[0]
    # model.load_state_dict(torch.load(file))
    # coeffs[file] = model.stepper.model.get_params_as_vector()

with open(snapshot_dir + '/' + root_name + '.pkl','wb') as f:
    pickle.dump(coeffs, f)