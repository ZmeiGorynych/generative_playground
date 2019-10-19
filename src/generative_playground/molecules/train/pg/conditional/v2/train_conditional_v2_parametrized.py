import argparse
import sys
if '/home/ubuntu/shared/GitHub' in sys.path:
    sys.path.remove('/home/ubuntu/shared/GitHub')
try:
    import generative_playground
except:
    import sys, os, inspect

    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../../../../../..')
    # sys.path.append('../../../../DeepRL')
    # sys.path.append('../../../../../transformer_pytorch')

# from deep_rl import *
# from generative_playground.models.problem.rl.network_heads import CategoricalActorCriticNet
# from generative_playground.train.rl.run_iterations import run_iterations
from generative_playground.molecules.rdkit_utils.rdkit_utils import num_atoms, num_aromatic_rings, NormalizedScorer
# from generative_playground.models.problem.rl.DeepRL_wrappers import BodyAdapter, MyA2CAgent
from generative_playground.molecules.model_settings import get_settings
from generative_playground.molecules.train.pg.hypergraph.main_train_policy_gradient_minimal import train_policy_gradient
from generative_playground.codec.hypergraph_grammar import GrammarInitializer
from generative_playground.molecules.guacamol_utils import guacamol_goal_scoring_functions, version_name_list
from generative_playground.models.temperature_schedule import toothy_exp_schedule, shifted_cosine_schedule, \
    reverse_toothy_exp_schedule, seesaw_exp_schedule

parser = argparse.ArgumentParser(description='Run simple model against guac')
parser.add_argument('objective', type=int, help="Guacamol objective index to target")
parser.add_argument('--attempt', help="Attempt number (used for multiple runs)", default='')
parser.add_argument('--lr', help="learning rate", default='')
parser.add_argument('--entropy_wgt', help="weight of the entropy penalty", default='')

args = parser.parse_args()
lr_str = args.lr
if not lr_str:
    lr_str = '0.05'
lr = float(lr_str)

ew_str = args.entropy_wgt
if not ew_str:
    ew_str = '0.1'
entropy_wgt= float(ew_str)

# num_batches = 30
batch_size = 30# 20 # was 75 but that was too much for a p2.xlarge
drop_rate = 0.5
molecules = True
grammar_cache = 'hyper_grammar_guac_10k_with_clique_collapse.pickle'#'hyper_grammar.pickle'
grammar = 'hypergraph:' + grammar_cache
settings = get_settings(molecules, grammar)
ver = 'v2'
obj_num = args.objective
reward_funs = guacamol_goal_scoring_functions(ver)
# this accepts a list of SMILES strings
reward_fun = reward_funs[obj_num]
# # later will run this ahead of time
# gi = GrammarInitializer(grammar_cache)
attempt = '_' + args.attempt if args.attempt else ''
# 'bench8obj' +
root_name = 'Ascope' + str(obj_num) + '_' + ver + '_lr_' + lr_str + '_ew_' + ew_str +'_' + attempt
max_steps = 60
model, gen_fitter, disc_fitter = train_policy_gradient(molecules,
                                                       grammar,
                                                       EPOCHS=100,
                                                       BATCH_SIZE=batch_size,
                                                       reward_fun_on=reward_fun,
                                                       max_steps=max_steps,
                                                       lr_on=lr,
                                                       # lr_schedule=shifted_cosine_schedule,
                                                       lr_discrim=0.0,
                                                       discrim_wt=0.0,
                                                       p_thresh=-10,
                                                       drop_rate=drop_rate,
                                                       reward_sm=0.0,
                                                       decoder_type='graph_conditional',  #'rnn_graph',# 'attention',
                                                       plot_prefix='',
                                                       dashboard=root_name,  # 'policy gradient',
                                                       save_file_root_name=root_name,
                                                       preload_file_root_name=root_name,  #'guacamol_ar_emb_node_rpev2_0lr2e-5',#'guacamol_ar_nodev2_0lr2e-5',#root_name,
                                                       smiles_save_file=root_name.replace(' ', '_') + '_smiles.zip',
                                                       on_policy_loss_type='advantage_record',
                                                       # rule_temperature_schedule=toothy_exp_schedule,
                                                       eps=0.0,
                                                       priors='conditional',
                                                       entropy_wgt=entropy_wgt)
# preload_file='policy_gradient_run.h5')
# for _ in range(num_batches):
while True:
    next(gen_fitter)
    # break
    # for _ in range(1):
    #     next(disc_fitter)
