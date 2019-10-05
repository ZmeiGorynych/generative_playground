import random
import sys, os, inspect
import argparse
import pickle
if '/home/ubuntu/shared/GitHub' in sys.path:
    sys.path.remove('/home/ubuntu/shared/GitHub')
from generative_playground.models.pg_runner import PolicyGradientRunner
from generative_playground.models.problem.genetic.genetic_opt import populate_data_cache, pick_model_to_run, \
    pick_model_for_crossover, generate_root_name
from generative_playground.models.problem.genetic.crossover import mutate, crossover
from generative_playground.molecules.guacamol_utils import guacamol_goal_scoring_functions
import networkx as nx


parser = argparse.ArgumentParser(description='Run simple model against guac')
parser.add_argument('objective', type=int, help="Guacamol objective index to target")
parser.add_argument('--attempt', help="Attempt number (used for multiple runs)", default='')
parser.add_argument('--lr', help="learning rate", default='')
parser.add_argument('--entropy_wgt', help="weight of the entropy penalty", default='')

args = parser.parse_args()
lr_str = args.lr
if not lr_str:
    lr_str = '0.02'
lr = float(lr_str)

ew_str = args.entropy_wgt
if not ew_str:
    ew_str = '0.0'
entropy_wgt= float(ew_str)


my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
snapshot_dir = os.path.realpath(my_location + '/data')
top_N = 10
p_mutate = 0.2
p_crossover = 0.2
num_batches = 100
batch_size = 50
relationships = nx.DiGraph()
grammar_cache = 'hyper_grammar_guac_10k_with_clique_collapse.pickle'  # 'hyper_grammar.pickle'
grammar = 'hypergraph:' + grammar_cache

ver = 'v2'
obj_num = args.objective
reward_funs = guacamol_goal_scoring_functions(ver)
reward_fun = reward_funs[obj_num]

attempt = '_' + args.attempt if args.attempt else ''
root_name ='a1genetic' + str(obj_num) + '_' + ver + '_lr_' + lr_str + '_ew_' + ew_str
snapshot_dir += '/' + root_name
if not os.path.isdir(snapshot_dir):
    os.mkdir(snapshot_dir)
runner_factory = lambda: PolicyGradientRunner(grammar,
                                              BATCH_SIZE=batch_size,
                                              reward_fun=reward_fun,
                                              max_steps=60,
                                              num_batches=num_batches,
                                              lr=lr,
                                              entropy_wgt=entropy_wgt,
                                              # lr_schedule=shifted_cosine_schedule,
                                              root_name=root_name,
                                              preload_file_root_name=None,
                                              plot_metrics=True,
                                              save_location=snapshot_dir,
                                              metric_smooth=0.0,
                                              decoder_type='graph_conditional',  # 'rnn_graph',# 'attention',
                                              on_policy_loss_type='advantage_record',
                                              rule_temperature_schedule=None,
                                              # lambda x: toothy_exp_schedule(x, scale=num_batches),
                                              eps=0.0,
                                              priors='conditional',
                                              )
data_cache = {}

while True:
    data_cache = populate_data_cache(snapshot_dir, data_cache)
    model = pick_model_to_run(data_cache, runner_factory, PolicyGradientRunner, snapshot_dir)

    orig_name = model.root_name
    model.set_root_name(generate_root_name(orig_name, data_cache))
    relationships.add_edge(orig_name, model.root_name)

    test = random.random()
    if test < p_mutate:
        model = mutate(model)
    elif test < p_mutate + p_crossover and len(data_cache) > 1:
        second_model = pick_model_for_crossover(data_cache, model, PolicyGradientRunner, snapshot_dir)
        model = crossover(model, second_model)
        relationships.add_edge(second_model.root_name, model.root_name)

    with open(snapshot_dir + '/' + model.root_name + '_lineage.pkl', 'wb') as f:
        pickle.dump(relationships, f)
    model.run()


