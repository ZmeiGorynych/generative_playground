import numpy as np
import pickle
import random
import sys

import torch
mp = torch.multiprocessing.get_context('forkserver')

if '/home/ubuntu/shared/GitHub' in sys.path:
    sys.path.remove('/home/ubuntu/shared/GitHub')
from generative_playground.models.pg_runner import PolicyGradientRunner
from generative_playground.models.problem.genetic.genetic_opt import populate_data_cache, pick_model_to_run, \
    pick_model_for_crossover, generate_root_name, extract_best
from generative_playground.models.problem.genetic.crossover import mutate, classic_crossover
from generative_playground.molecules.guacamol_utils import guacamol_goal_scoring_functions
from generative_playground.utils.visdom_helper import Dashboard
from generative_playground.models.param_sampler import ParameterSampler, extract_params_rewards

import networkx as nx


def run_model(queue, root_name, run_index, save_location):
    print('Running: {}'.format(run_index))
    model = PolicyGradientRunner.load_from_root_name(save_location, root_name)
    model.run()
    queue.put((run_index, model.root_name,))


def run_genetic_opt(top_N=10,
                    p_mutate=0.2,
                    mutate_num_best=64,
                    mutate_use_total_probs=False,
                    p_crossover=0.2,
                    num_batches=100,
                    batch_size=30,
                    snapshot_dir=None,
                    entropy_wgt=0.0,
                    root_name=None,
                    obj_num=None,
                    ver='v2',
                    lr=0.01,
                    num_runs=100,
                    num_explore=5,
                    plot_single_runs=True,
                    steps_with_no_improvement=10,
                    reward_aggregation=np.median,
                    attempt='',  # only used for disambiguating plotting
                    max_steps=90,
                    past_runs_graph_file=None
                    ):

    manager = mp.Manager()
    queue = manager.Queue()

    relationships = nx.DiGraph()
    grammar_cache = 'hyper_grammar_guac_10k_with_clique_collapse.pickle'  # 'hyper_grammar.pickle'
    grammar = 'hypergraph:' + grammar_cache

    reward_funs = guacamol_goal_scoring_functions(ver)
    reward_fun = reward_funs[obj_num]

    split_name = root_name.split('_')
    split_name[0] += 'Stats'
    dash_name = '_'.join(split_name) + attempt
    vis = Dashboard(dash_name, call_every=1)

    first_runner_factory = lambda: PolicyGradientRunner(grammar,
                                                        BATCH_SIZE=batch_size,
                                                        reward_fun=reward_fun,
                                                        max_steps=max_steps,
                                                        num_batches=num_batches,
                                                        lr=lr,
                                                        entropy_wgt=entropy_wgt,
                                                        # lr_schedule=shifted_cosine_schedule,
                                                        root_name=root_name,
                                                        preload_file_root_name=None,
                                                        plot_metrics=plot_single_runs,
                                                        save_location=snapshot_dir,
                                                        metric_smooth=0.0,
                                                        decoder_type='graph_conditional_sparse',
                                                        # 'graph_conditional',  # 'rnn_graph',# 'attention',
                                                        on_policy_loss_type='advantage_record',
                                                        rule_temperature_schedule=None,
                                                        # lambda x: toothy_exp_schedule(x, scale=num_batches),
                                                        eps=0.0,
                                                        priors='conditional',
                                                        )

    init_thresh = 50
    pca_dim = 10
    if past_runs_graph_file:
        params, rewards = extract_params_rewards(past_runs_graph_file)
        sampler = ParameterSampler(params, rewards, init_thresh=init_thresh, pca_dim=pca_dim)
    else:
        sampler = None
    data_cache = {}
    best_so_far = float('-inf')
    steps_since_best = 0

    initial = True
    should_stop = False
    run = 0

    with mp.Pool(4) as p:
        while not should_stop:
            data_cache = populate_data_cache(snapshot_dir, data_cache)
            if run < num_explore:
                model = first_runner_factory()
                if sampler:
                    model.params = sampler.sample()
            else:
                model = (
                    pick_model_to_run(
                        data_cache,
                        PolicyGradientRunner,
                        snapshot_dir,
                        num_best=top_N
                    )
                    if data_cache else
                    first_runner_factory()
                )

            orig_name = model.root_name
            model.set_root_name(generate_root_name(orig_name, data_cache))

            if run > num_explore:
                relationships.add_edge(orig_name, model.root_name)

                if random.random() < p_crossover and len(data_cache) > 1:
                    second_model = pick_model_for_crossover(
                        data_cache,
                        model,
                        PolicyGradientRunner,
                        snapshot_dir
                    )
                    model = classic_crossover(model, second_model)
                    relationships.add_edge(second_model.root_name, model.root_name)

                if random.random() < p_mutate:
                    model = mutate(
                        model,
                        pick_best=mutate_num_best,
                        total_probs=mutate_use_total_probs
                    )
                    relationships.node[model.root_name]['mutated'] = True
                else:
                    relationships.node[model.root_name]['mutated'] = False

                with open(
                    snapshot_dir + '/' + model.root_name + '_lineage.pkl', 'wb'
                ) as f:
                    pickle.dump(relationships, f)

            model.save()

            if initial is True:
                for _ in range(4):
                    print('Starting {}'.format(run))
                    p.apply_async(
                        run_model,
                        (queue, model.root_name, run, snapshot_dir)
                    )
                    run += 1
                initial = False
            else:
                print('Starting {}'.format(run))
                p.apply_async(
                    run_model,
                    (queue, model.root_name, run, snapshot_dir)
                )
                run += 1

            finished_run, finished_root_name = queue.get(block=True)
            print('Finished: {}'.format(finished_root_name))

            data_cache = populate_data_cache(snapshot_dir, data_cache)
            my_rewards = data_cache[finished_root_name]['best_rewards']
            metrics = {'max': my_rewards.max(), 'median': np.median(my_rewards), 'min': my_rewards.min()}
            metric_dict = {
                'type': 'line',
                'X': np.array([finished_run]),
                'Y': np.array([[val for key, val in metrics.items()]]),
                'opts': {'legend': [key for key, val in metrics.items()]}
            }

            vis.plot_metric_dict({'worker rewards': metric_dict})

            this_agg_reward = reward_aggregation(my_rewards)
            if this_agg_reward > best_so_far:
                best_so_far = this_agg_reward
                steps_since_best = 0
            else:
                steps_since_best += 1

            should_stop = (
                steps_since_best >= steps_with_no_improvement
                and finished_run > num_explore + steps_with_no_improvement
            )

        p.terminate()

    return extract_best(data_cache, 1)


def run_initial_scan(num_batches=100,
                     batch_size=30,
                     snapshot_dir=None,
                     entropy_wgt=0.0,
                     root_name=None,
                     obj_num=None,
                     ver='v2',
                     lr=0.01,
                     attempt='',
                     plot=False
                     ):
    grammar_cache = 'hyper_grammar_guac_10k_with_clique_collapse.pickle'  # 'hyper_grammar.pickle'
    grammar = 'hypergraph:' + grammar_cache
    reward_funs = guacamol_goal_scoring_functions(ver)
    reward_fun = reward_funs[obj_num]

    first_runner = lambda: PolicyGradientRunner(grammar,
                                                BATCH_SIZE=batch_size,
                                                reward_fun=reward_fun,
                                                max_steps=60,
                                                num_batches=num_batches,
                                                lr=lr,
                                                entropy_wgt=entropy_wgt,
                                                # lr_schedule=shifted_cosine_schedule,
                                                root_name=root_name,
                                                preload_file_root_name=None,
                                                plot_metrics=plot,
                                                save_location=snapshot_dir,
                                                metric_smooth=0.0,
                                                decoder_type='graph_conditional',  # 'rnn_graph',# 'attention',
                                                on_policy_loss_type='advantage_record',
                                                rule_temperature_schedule=None,
                                                # lambda x: toothy_exp_schedule(x, scale=num_batches),
                                                eps=0.0,
                                                priors='conditional',
                                                )

    run = 0
    while True:
        model = first_runner()
        orig_name = model.root_name
        model.set_root_name(generate_root_name(orig_name, {}))
        model.run()
