import sys
try:
    sys.path.remove('/home/ubuntu/shared/GitHub')
except:
    pass
import os, inspect
my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append('../../../..')
from generative_playground.models.problem.mcts.mcts import run_mcts
from generative_playground.models.problem.mcts.node import MCTSNodeLocalThompson, MCTSNodeGlobalThompson
import argparse

try:
    parser = argparse.ArgumentParser(description='Run simple model against guac')
    parser.add_argument('objective', type=int, help="Guacamol objective index to target")
    parser.add_argument('--attempt', help="Attempt number (used for multiple runs)", default='')

    args = parser.parse_args()
    obj_num = args.objective
except:
    obj_num = 0


run_mcts(kind='model_mixed',
         compress_data_store=True,
         base_name='MCTS_mixed_model',
         obj_num=obj_num,
         ver='v2',
         reset_cache=True,
         decay=0.95,
         updates_to_refresh=10,
         batch_size=20)