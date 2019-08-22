from generative_playground.utils.batch_run import batch_run
import os

here = os.path.realpath(__file__)
ohio = False#True
file = 'mcts_local'#'conditional'
source_root = "/home/ubuntu/shared/GitHub/generative_playground/src"
train_root = source_root + "/generative_playground/molecules/train"

if file=='conditional':
    python_file = '{}/pg/conditional/v2/train_conditional_v2_parametrized.py'.format(train_root)
elif file == 'mcts_global':
    python_file = '{}/mcts/run_global_model_mcts.py'.format(train_root)
elif file=='mcts_local':
    python_file = '{}/mcts/run_local_model_mcts.py'.format(train_root)

if ohio:
    key_file = os.path.realpath("../../../../../../../aws_ohio.pem")
else:
    key_file = os.path.realpath("../../../../../../../aws_second_key_pair.pem")

job_assignments = {}
# ips = ['34.245.235.9',
#        '54.171.208.205',
#        '34.243.135.222',
#        '34.247.84.50',
#        '54.229.87.2']
# for i, ip in enumerate(ips):
#     job_assignments[ip] = [4*i+j for j in range(4)]
#
# redo = [0]
# job_assignments = {key: [v for v in value if v in redo] for key, value in job_assignments.items()}
# job_assignments = {key: value for key, value in job_assignments.items() if value }
# job_assignments = {'3.15.182.107': [0]}
job_assignments = {'34.254.195.73': [0]}
batch_run(source_root, python_file, key_file, job_assignments)

# screen -ls
# screen -r ...
# ^a d