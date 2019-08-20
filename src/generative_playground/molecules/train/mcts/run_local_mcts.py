from generative_playground.models.problem.mcts.mcts import run_mcts
from generative_playground.models.problem.mcts.node import MCTSNodeLocalThompson, MCTSNodeGlobalThompson

run_mcts(num_batches=10000,
         node_type=MCTSNodeLocalThompson,
         compress_data_store=True, # 50% faster but really disk-hungry
         base_name='MCTS_local',
         ver='v2',
         obj_num=0)