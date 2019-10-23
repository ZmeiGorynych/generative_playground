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
from generative_playground.molecules.train.genetic.main_genetic_train import run_genetic_opt, run_initial_scan

parser = argparse.ArgumentParser(description='Run simple model against guac')
parser.add_argument('objective', type=int, help="Guacamol objective index to target")
parser.add_argument('--attempt', help="Attempt number (used for multiple runs)", default='')
parser.add_argument('--lr', help="learning rate", default='')
parser.add_argument('--entropy_wgt', help="weight of the entropy penalty", default='')

args = parser.parse_args()
lr_str = args.lr
if not lr_str:
    lr_str = '0.1'
lr = float(lr_str)

ew_str = args.entropy_wgt
if not ew_str:
    ew_str = '0.1'
entropy_wgt= float(ew_str)


my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
snapshot_dir = os.path.realpath(my_location + '/data')

# attempt = args.attempt if args.attempt else ''
obj_num = args.objective
ver = 'v2'
root_name = 'zzzscan' + str(obj_num) + '_' + ver + '_lr' + lr_str + '_ew' + ew_str
snapshot_dir += '/' + root_name
if not os.path.isdir(snapshot_dir):
    os.mkdir(snapshot_dir)

num_batches = 400

best = run_initial_scan(num_batches = num_batches,
                    batch_size = 30,
                    snapshot_dir=snapshot_dir,
                    entropy_wgt =entropy_wgt*lr*10,
                    root_name = root_name,
                       attempt = args.attempt,
                    obj_num=obj_num,
                    ver=ver,
                    lr=lr,
                        plot=True
                    )
print(best)