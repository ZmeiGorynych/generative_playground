import multiprocessing as mp
import random, math
import sys, os, inspect
import argparse
import pickle
import numpy as np
from hyperopt import STATUS_OK, hp, Trials, fmin
from hyperopt.mongoexp import MongoTrials
from hyperopt.pyll.stochastic import sample
from timeit import default_timer as timer
from hyperopt import tpe
from pymongo import MongoClient


if '/home/ubuntu/shared/GitHub' in sys.path:
    sys.path.remove('/home/ubuntu/shared/GitHub')

from generative_playground.molecules.train.genetic.parallel_genetic_train import run_genetic_opt



def try_for_params(params: dict,
                   snapshot_dir: str,
                   run_name: str = 'test',
                   ver='v2',
                   obj_num=0,
                   ):
    lr = params.get('lr',0.05)
    lr_str = str(lr)[:5]
    entropy_wgt = 0.0


    # snapshot_dir = os.path.realpath(my_location + '/data')

    attempt = ''
    obj_num = obj_num
    ver = ver
    past_runs_graph_file = None#snapshot_dir + '/geneticA' + str(obj_num) + '_graph.zip'
    root_name = run_name + str(obj_num) + '_' + ver + '_lr' + lr_str
    snapshot_dir += '/' + root_name
    if not os.path.isdir(snapshot_dir):
        os.mkdir(snapshot_dir)

    max_steps = 90
    batch_size = 8

    top_N = 16
    p_mutate = params.get('p_mutate', 0.2)
    mutate_num_best = 16
    p_crossover = params.get('p_crossover', 0.5)
    num_batches = 4
    mutate_use_total_probs = True
    num_explore=0

    mp.freeze_support()
    best = run_genetic_opt(top_N=top_N,
                           p_mutate=p_mutate,
                           mutate_num_best=mutate_num_best,
                           mutate_use_total_probs=mutate_use_total_probs,
                           p_crossover=p_crossover,
                           num_batches=num_batches,
                           batch_size=batch_size, # 30
                           max_steps=max_steps,
                           snapshot_dir=snapshot_dir,
                           entropy_wgt=entropy_wgt,
                           root_name=root_name,
                           obj_num=obj_num,
                           ver=ver,
                           lr=lr,
                           num_runs=1000,
                           plot_single_runs=True,
                           attempt=attempt,
                           past_runs_graph_file=past_runs_graph_file,
                           num_explore=num_explore
                           )

    fun_vec = list(best.values())[0]['best_rewards']
    loss = -fun_vec.max()
    loss_variance = fun_vec.var()

    return {'loss': loss, 'loss_variance': loss_variance, 'status': STATUS_OK}

if __name__ == '__main__':
    space = {'lr': hp.loguniform('lr', math.log(0.001), math.log(0.1)),
             'p_mutate': hp.uniform('p_mutate', 0.0, 1.0),
             'p_crossover': hp.uniform('p_crossover', 0.0, 1.0),
             }
    mongo_server = '52.213.134.161:27017'
    tpe_algorithm = tpe.suggest
    run_name = 'HyperTest'
    client = MongoClient('mongodb://' + mongo_server + '/')
    client.drop_database(run_name)
    mtrials = MongoTrials('mongo://' + mongo_server + '/' + run_name + '/jobs', exp_key='exp1')
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    snapshot_dir = '/home/ubuntu/data/'

    objective = lambda params: try_for_params(params,
                                              run_name=run_name,
                                              ver='v2',
                                              obj_num=0,
                                              snapshot_dir=snapshot_dir)
    mbest = fmin(fn=objective,
                 space=space,
                 algo=tpe_algorithm,
                 max_evals=10,
                 trials=mtrials,
                 rstate=np.random.RandomState(50))

    mtrials_results = sorted(mtrials.results, key=lambda x: x['loss'])
    print(mtrials_results[:10])
