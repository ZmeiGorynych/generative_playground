import sys, os, inspect

if '/home/ubuntu/shared/GitHub' in sys.path:
    sys.path.remove('/home/ubuntu/shared/GitHub')
try:
    import generative_playground
except:
    import sys, os, inspect

    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # sys.path.append('../../../../../..')
    # sys.path.append('../../../../DeepRL')
    # sys.path.append('../../../../../transformer_pytorch')

# from deep_rl import *
# from generative_playground.models.problem.rl.network_heads import CategoricalActorCriticNet
# from generative_playground.train.rl.run_iterations import run_iterations
# from generative_playground.models.problem.rl.DeepRL_wrappers import BodyAdapter, MyA2CAgent
from generative_playground.molecules.model_settings import get_settings
from generative_playground.molecules.train.pg.hypergraph.main_train_policy_gradient_minimal import train_policy_gradient
from generative_playground.molecules.guacamol_utils import guacamol_goal_scoring_functions
from generative_playground.models.temperature_schedule import toothy_exp_schedule, shifted_cosine_schedule, \
    seesaw_exp_schedule
from generative_playground.models.pg_runner import PolicyGradientRunner

batch_size = 20  # 20
num_batches = 5
grammar_cache = 'hyper_grammar_guac_10k_with_clique_collapse.pickle'  # 'hyper_grammar.pickle'
grammar = 'hypergraph:' + grammar_cache
ver = 'v2'
obj_num = 0
reward_funs = guacamol_goal_scoring_functions(ver)
# this accepts a list of SMILES strings
reward_fun = reward_funs[obj_num]

root_name = 'xtest9' + ver + '_' + str(obj_num) + '_lr0.02'
max_steps = 60
root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
root_location = root_location + '/../../../'
save_location = os.path.realpath(root_location + 'pretrained/')


runner = PolicyGradientRunner(grammar,
                              BATCH_SIZE=batch_size,
                              reward_fun=reward_fun,
                              max_steps=max_steps,
                              num_batches=num_batches,
                              lr=0.02,
                              entropy_wgt=0.1,
                              # lr_schedule=shifted_cosine_schedule,
                              root_name=root_name,
                              preload_file_root_name=None,
                              plot_metrics=True,
                              save_location=save_location,
                              metric_smooth=0.0,
                              decoder_type='graph_conditional',  # 'rnn_graph',# 'attention',
                              on_policy_loss_type='advantage_record',
                              rule_temperature_schedule=lambda x: toothy_exp_schedule(x, scale=num_batches),
                              eps=0.0,
                              priors='conditional',
                              )
# preload_file='policy_gradient_run.h5')

runner.set_root_name('whatever')
save_fn = runner.run()
runner = PolicyGradientRunner.load(save_fn)
runner.set_root_name('whatever2')
runner.run()
