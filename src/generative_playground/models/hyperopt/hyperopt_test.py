import math
import numpy as np
from hyperopt import STATUS_OK, hp, Trials, fmin
from hyperopt.mongoexp import MongoTrials
from hyperopt.pyll.stochastic import sample
from timeit import default_timer as timer
from hyperopt import tpe
from generative_playground.models.hyperopt.worker import spawn_hyperopt_worker
import subprocess

from pymongo import MongoClient

def objective(params):
    start = timer()
    loss = math.sin(params['x1']) * math.cos(params['x2']) + params['x2'] + params['a'] * params['b']
    run_time = timer() - start
    return {'loss': loss, 'train_time': run_time, 'status': STATUS_OK}


space = {'x1': hp.uniform('x1', 1, 5),
         'x2': hp.uniform('x2', 1, 8),
         'a': hp.quniform('a', 1, 3, 1),
         'b': hp.choice('b', [1, 5])
         }

# optimization algorithm
tpe_algorithm = tpe.suggest

## first do a local opt
bayes_trials = Trials()

# Run optimization
best = fmin(fn=objective,
            space=space,
            algo=tpe_algorithm,
            max_evals=10,
            trials=bayes_trials,
            rstate=np.random.RandomState(50))

# Sort the trials with lowest loss (highest AUC) first
bayes_trials_results = sorted(bayes_trials.results, key=lambda x: x['loss'])
print(bayes_trials_results[:10])
mongo_server = '52.213.134.161:27017'
client = MongoClient('mongodb://' + mongo_server + '/')
db_to_use = 'test_db'
client.drop_database(db_to_use)

spawn_hyperopt_worker(mongo_server, db_to_use)

## now try a distributed opt via mongo
mtrials = MongoTrials('mongo://' + mongo_server + '/' + db_to_use + '/jobs',
                      exp_key='exp1')
# Run optimization
mbest = fmin(fn=objective,
            space=space,
            algo=tpe_algorithm,
            max_evals=30,
            trials=mtrials,
            rstate=np.random.RandomState(50))

print('done!')
