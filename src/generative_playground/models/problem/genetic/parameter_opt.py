import math, os, inspect
import numpy as np
from hyperopt import STATUS_OK, hp, Trials, fmin
from hyperopt.pyll.stochastic import sample
from timeit import default_timer as timer
from hyperopt import tpe
from generative_playground.molecules.train.genetic.main_genetic_train import run_genetic_opt
from hyperopt.mongoexp import MongoTrials


my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
base_snapshot_dir = os.path.realpath(my_location + '../../../../molecules/train/genetic/data')


# start 16 processes, run each until 'saturation', that being the flattening of the 90th percentile
# from all 16, get the mean and distribution of all unconditional and conditional probabilities, filtered by what's allowed
# method to get and set the probabilities vectors
# now sample the initial population from that distribution
# run genetic opt from that sample - each 'run' returns a vector of best values achieved
# modify the comparator function to do TS from these
# determine the 'typical' number of steps to run the go from plotting the mean, std of that vector by the go algo
# wrap this as one unit of optimization
# result returned is the mean, std of the best distribution in the population

def objective(params, base_name, obj_num=7):
    start = timer()
    top_N = 10
    num_batches = 10
    lr = 0.01
    ver = 'v2'
    p_mutate = params['p_mutate']
    p_crossover = params['p_crossover']
    entropy_wgt = params['entropy_wgt']
    root_name = base_name + ver + '_' + str(obj_num) \
                + '_pm' + str(p_mutate)[:4] + '_pc' + str(p_crossover)[:4]
                # + '_lr' + str(lr)[:5] + '_ew' + str(entropy_wgt)[:4] \
    snapshot_dir = base_snapshot_dir + '/' + root_name
    if not os.path.isdir(snapshot_dir):
        os.mkdir(snapshot_dir)

    best_result = run_genetic_opt(top_N=top_N,
                           p_mutate=p_mutate,
                           p_crossover=p_crossover,
                           num_batches=num_batches,
                           batch_size=30,
                           snapshot_dir=snapshot_dir,
                           entropy_wgt=entropy_wgt,
                           root_name=root_name,
                           obj_num=obj_num,
                           ver=ver,
                           lr=lr,
                           num_runs=1,
                           steps_with_no_improvement=3,
                           plot_single_runs=True
                           )

    best = list(best_result.values())[0]['best_rewards']

    run_time = timer() - start
    return {'loss': -np.median(best),
            'loss_variance': np.var(best),
            'train_time': run_time,
            'status': STATUS_OK}


space = {'p_mutate': hp.uniform('p_mutate', 0,1),
         'p_crossover': hp.uniform('p_crossover', 0,1),
         'entropy_wgt': hp.loguniform('entropy_wgt', np.log(0.001), np.log(1))
         }

# optimization algorithm
tpe_algorithm = tpe.suggest
trials = Trials()
# trials = MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp1')
# hyperopt-mongo-worker --mongo=localhost:1234/foo_db --poll-interval=0.1
# Run optimization
best = fmin(fn=lambda x: objective(x, base_name='tpe_test2', obj_num=8), space=space, algo=tpe.suggest,
            max_evals=1000, trials=trials, rstate=np.random.RandomState(50))
