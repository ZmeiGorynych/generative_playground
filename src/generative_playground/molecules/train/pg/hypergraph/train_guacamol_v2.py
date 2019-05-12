try:
    import generative_playground
except:
    import sys, os, inspect

    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../../../../..')
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



batch_size = 20 # 20
drop_rate = 0.5
molecules = True
grammar_cache = 'hyper_grammar.pickle'
grammar = 'hypergraph:' + grammar_cache
settings = get_settings(molecules, grammar)
ver = 'v2'
obj_num = 0
reward_funs = guacamol_goal_scoring_functions(ver)
reward_fun = reward_funs[obj_num]
# later will run this ahead of time
gi = GrammarInitializer(grammar_cache)
# if True:
#     gi.delete_cache()
#     gi = GrammarInitializer(grammar_cache)
#     max_steps_smiles = gi.init_grammar(1000)

root_name = 'guacamol_ar_node' + ver + '_' + str(obj_num) + 'lr2e-5'
max_steps = 50
model, gen_fitter, disc_fitter = train_policy_gradient(molecules,
                                                       grammar,
                                                       EPOCHS=100,
                                                       BATCH_SIZE=batch_size,
                                                       reward_fun_on=reward_fun,
                                                       max_steps=max_steps,
                                                       lr_on=2e-5,
                                                       lr_discrim=5e-4,
                                                       discrim_wt=0.0,
                                                       p_thresh=-10,
                                                       drop_rate=drop_rate,
                                                       reward_sm=0.0,
                                                       decoder_type='attn_graph_node',  #'rnn_graph',# 'attention',
                                                       plot_prefix='',
                                                       dashboard= root_name,  # 'policy gradient',
                                                       save_file_root_name=root_name,
                                                       preload_file_root_name=None,#root_name,
                                                       smiles_save_file=None,  # 'pg_smiles_hg1.h5',
                                                       on_policy_loss_type='advantage_record')
# preload_file='policy_gradient_run.h5')

while True:
    next(gen_fitter)
    # for _ in range(1):
    #     next(disc_fitter)
