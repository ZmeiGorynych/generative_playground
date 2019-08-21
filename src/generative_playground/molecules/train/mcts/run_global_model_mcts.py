from generative_playground.models.problem.mcts.mcts import run_mcts
from generative_playground.models.problem.mcts.node import MCTSNodeLocalThompson, MCTSNodeGlobalThompson
import argparse
parser = argparse.ArgumentParser(description='Run simple model against guac')
parser.add_argument('objective', type=int, help="Guacamol objective index to target")
parser.add_argument('--attempt', help="Attempt number (used for multiple runs)", default='')

args = parser.parse_args()

run_mcts(num_batches=10000,
         kind='model_global',
         compress_data_store=True,
         base_name='MCTS_global_model',
         obj_num=args.objective,
            ver='trivial' )