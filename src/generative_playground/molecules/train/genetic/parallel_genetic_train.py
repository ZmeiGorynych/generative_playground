import multiprocessing as mp
import pickle
import random

import networkx as nx
import numpy as np
from generative_playground.models.param_sampler import (
    ParameterSampler,
    extract_params_rewards
)
from generative_playground.models.pg_runner import PolicyGradientRunner
from generative_playground.models.problem.genetic.crossover import (
    mutate,
    classic_crossover,
)
from generative_playground.models.problem.genetic.genetic_opt import (
    populate_data_cache,
    pick_model_to_run,
    pick_model_for_crossover,
    generate_root_name,
    extract_best,
)
from generative_playground.molecules.guacamol_utils import (
    guacamol_goal_scoring_functions,
)
from generative_playground.utils.visdom_helper import Dashboard


def main(job_id, params):
    runs = 0
    with mp.Pool(4) as p:
        pass

        if this_agg_reward > best_so_far:
            best_so_far = this_agg_reward
            steps_since_best = 0
        else:
            steps_since_best += 1

        should_stop = (
            steps_since_best >= self.steps_with_no_improvement
            and run > self.num_explore + self.steps_with_no_improvement
        )
        if should_stop:
            break


class ParallelGeneticOptWorker:
    def __init__(
        self,
        top_N=10,
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
        past_runs_graph_file=None,
    ):
        self.relationships = nx.DiGraph()
        self.grammar_cache = 'hyper_grammar_guac_10k_with_clique_collapse.pickle'
        self.grammar = 'hypergraph:' + self.grammar_cache

        self.mutate_num_best = mutate_num_best
        self.mutate_use_total_probs = mutate_use_total_probs
        self.num_explore = num_explore
        self.p_crossover = p_crossover
        self.p_mutate = p_mutate
        self.reward_aggregation = reward_aggregation
        self.steps_with_no_improvement = steps_with_no_improvement
        self.top_N = top_N

        reward_funs = guacamol_goal_scoring_functions(ver)
        self.reward_fun = reward_funs[obj_num]

        split_name = root_name.split('_')
        split_name[0] += 'Stats'
        dash_name = '_'.join(split_name) + attempt
        self.vis = Dashboard(dash_name, call_every=1)

        self.first_runner_factory = lambda: PolicyGradientRunner(
            self.grammar,
            BATCH_SIZE=batch_size,
            reward_fun=self.reward_fun,
            max_steps=max_steps,
            num_batches=num_batches,
            lr=lr,
            entropy_wgt=entropy_wgt,
            root_name=root_name,
            preload_file_root_name=None,
            plot_metrics=plot_single_runs,
            save_location=snapshot_dir,
            metric_smooth=0.0,
            decoder_type='graph_conditional_sparse',
            on_policy_loss_type='advantage_record',
            rule_temperature_schedule=None,
            eps=0.0,
            priors='conditional',
        )

        init_thresh = 50
        pca_dim = 10
        self.sampler = None
        if past_runs_graph_file:
            params, rewards = extract_params_rewards(past_runs_graph_file)
            self.sampler = ParameterSampler(
                params,
                rewards,
                init_thresh=init_thresh,
                pca_dim=pca_dim
            )

        self.data_cache = {}
        self.best_so_far = float('-inf')
        self.steps_since_best = 0
        self.data_cache = populate_data_cache(self.snapshot_dir, self.data_cache)

    def run_explore(self):
        model = self.first_runner_factory()
        if self.sampler:
            model.params = self.sampler.sample()

    def run_normal(self):
        best_so_far = float('-inf')
        steps_since_best = 0

        for run in range(num_runs):
            model = (
                pick_model_to_run(
                    self.data_cache,
                    PolicyGradientRunner,
                    self.snapshot_dir,
                    num_best=self.top_N
                )
                if self.data_cache else
                self.first_runner_factory()
            )

            orig_name = model.root_name
            model.set_root_name(generate_root_name(orig_name, self.data_cache))

            self.relationships.add_edge(orig_name, model.root_name)

            if random.random() < self.p_crossover and len(self.data_cache) > 1:
                second_model = pick_model_for_crossover(
                    self.data_cache,
                    model,
                    PolicyGradientRunner,
                    self.snapshot_dir,
                )
                model = classic_crossover(model, second_model)
                self.relationships.add_edge(second_model.root_name, model.root_name)

            if random.random() < self.p_mutate:
                model = mutate(
                    model,
                    pick_best=self.mutate_num_best,
                    total_probs=self.mutate_use_total_probs,
                )
                self.relationships.node[model.root_name]['mutated'] = True
            else:
                self.relationships.node[model.root_name]['mutated'] = False

            with open(
                self.snapshot_dir + '/' + model.root_name + '_lineage.pkl', 'wb'
            ) as f:
                pickle.dump(self.relationships, f)

            model.run()
            self.data_cache = populate_data_cache(self.snapshot_dir, self.data_cache)
            my_rewards = self.data_cache[model.root_name]['best_rewards']
            metrics = {
                'max': my_rewards.max(),
                'median': np.median(my_rewards),
                'min': my_rewards.min()
            }
            metric_dict = {
                'type': 'line',
                'X': np.array([run]),
                'Y': np.array([[val for key, val in metrics.items()]]),
                'opts': {'legend': [key for key, val in metrics.items()]}
            }

            self.vis.plot_metric_dict({'worker rewards': metric_dict})

            this_agg_reward = self.reward_aggregation(my_rewards)
            if this_agg_reward > best_so_far:
                best_so_far = this_agg_reward
                steps_since_best = 0
            else:
                steps_since_best += 1

            should_stop = (
                steps_since_best >= self.steps_with_no_improvement
                and run > self.num_explore + self.steps_with_no_improvement
            )
            if should_stop:
                break

        return extract_best(self.data_cache, 1)

    def run_initial_scan(
        self,
        num_batches=100,
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
        grammar_cache = 'hyper_grammar_guac_10k_with_clique_collapse.pickle'
        grammar = 'hypergraph:' + grammar_cache
        reward_funs = guacamol_goal_scoring_functions(ver)
        reward_fun = reward_funs[obj_num]

        first_runner = lambda: PolicyGradientRunner(
            grammar,
            BATCH_SIZE=batch_size,
            reward_fun=reward_fun,
            max_steps=60,
            num_batches=num_batches,
            lr=lr,
            entropy_wgt=entropy_wgt,
            root_name=root_name,
            preload_file_root_name=None,
            plot_metrics=plot,
            save_location=snapshot_dir,
            metric_smooth=0.0,
            decoder_type='graph_conditional',
            on_policy_loss_type='advantage_record',
            rule_temperature_schedule=None,
            eps=0.0,
            priors='conditional',
        )

        while True:
            model = first_runner()
            orig_name = model.root_name
            model.set_root_name(generate_root_name(orig_name, {}))
            model.run()
