import sys
try:
    sys.path.remove('/home/ubuntu/shared/GitHub')
except:
    pass

from generative_playground.models.problem.mcts.mcts import run_mcts
from generative_playground.models.problem.mcts.node import MCTSNodeLocalThompson, MCTSNodeGlobalThompson
import argparse

try:
    parser = argparse.ArgumentParser(description='Run simple model against guac')
    parser.add_argument('objective', type=int, help="Guacamol objective index to target")
    parser.add_argument('--attempt', help="Attempt number (used for multiple runs)", default='')

    args = parser.parse_args()
    obj_num = args.objective
    attempt = args.attempt
except:
    obj_num = 0
    attempt = ''

ver = 'v2'
run_mcts(kind='model_global',
         compress_data_store=True,
         root_name='MCTSGlobalModel' + ver + '_obj' + str(obj_num) + '_attempt' + attempt,
         obj_num=obj_num,
         ver=ver,
         reset_cache=False,
         penalize_repetition=True,
         batch_size=5,
         num_batches=30,
         entropy_weight=0
         )
