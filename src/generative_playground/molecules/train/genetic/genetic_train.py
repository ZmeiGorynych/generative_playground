import random
import sys, os, inspect
import argparse
import pickle

if '/home/ubuntu/shared/GitHub' in sys.path:
    sys.path.remove('/home/ubuntu/shared/GitHub')

from generative_playground.molecules.train.genetic.parallel_genetic_train import run_genetic_opt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run simple model against guac')
    parser.add_argument('objective', type=int, help="Guacamol objective index to target")
    parser.add_argument('--attempt', help="Attempt number (used for multiple runs)", default='')
    parser.add_argument('--lr', help="learning rate", default='')
    parser.add_argument('--entropy_wgt', help="weight of the entropy penalty", default='')

    args = parser.parse_args()
    lr_str = args.lr
    if not lr_str:
        lr_str = '0.02'
    lr = float(lr_str)

    ew_str = args.entropy_wgt
    if not ew_str:
        ew_str = '0.0'
    entropy_wgt = float(ew_str)

    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # snapshot_dir = os.path.realpath(my_location + '/../../../../../../../../data')
    snapshot_dir = os.path.realpath(my_location + '/data')

    attempt = '_' + args.attempt if args.attempt else ''
    obj_num = args.objective
    ver = 'v2'
    past_runs_graph_file = None#snapshot_dir + '/geneticA' + str(obj_num) + '_graph.zip'
    root_name = 'geneticZZZZC' + str(obj_num) + '_' + ver + '_lr' + lr_str + '_ew' + ew_str
    snapshot_dir += '/' + root_name
    if not os.path.isdir(snapshot_dir):
        os.mkdir(snapshot_dir)

    max_steps = 75
    batch_size = 10

    top_N = 16
    p_mutate = 0.2
    mutate_num_best = 64
    p_crossover = 0.5
    num_batches = 4
    mutate_use_total_probs = True
    num_explore=0


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
    print(best)


