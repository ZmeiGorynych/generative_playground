import os, sys, inspect
my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append('../../../..')
from generative_playground.models.problem.mcts.mcts import run_mcts
from generative_playground.models.problem.mcts.node import MCTSNodeLocalThompson, MCTSNodeGlobalThompson

run_mcts(num_batches=10000,
         kind='thompson_local',
         compress_data_store=True, # not compressing is 50% faster but really disk-hungry
         base_name='MCTS_local',
         ver='v2',
         obj_num=0,
         reset_cache=True)