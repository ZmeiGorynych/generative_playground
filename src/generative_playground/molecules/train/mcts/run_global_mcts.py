from generative_playground.models.problem.mcts.mcts import run_mcts
from generative_playground.models.problem.mcts.node import MCTSNodeLocalThompson, MCTSNodeGlobalThompson

run_mcts(num_batches=10000,
         node_type=MCTSNodeGlobalThompson,
         compress_data_store=False,
         dashboard_name='MCTS_global')