import random
import numpy as np
import os, sys, inspect
from unittest import TestCase
from uuid import uuid4
from generative_playground.models.problem.genetic.genetic_opt import populate_data_cache, \
    pick_model_to_run, generate_root_name
from generative_playground.models.problem.genetic.crossover import crossover, mutate, classic_crossover, classic_mutate
from generative_playground.models.pg_runner import PolicyGradientRunner
from generative_playground.molecules.guacamol_utils import guacamol_goal_scoring_functions

my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
test_root = os.path.realpath(my_location)
test_data_dir = os.path.realpath(test_root + '/test_data')


class Dummy:
    @classmethod
    def load_from_root_name(cls, x):
        return x


grammar_cache = 'hyper_grammar_guac_10k_with_clique_collapse.pickle'  # 'hyper_grammar.pickle'
grammar = 'hypergraph:' + grammar_cache
ver = 'v2'
obj_num = 0
reward_funs = guacamol_goal_scoring_functions(ver)
# this accepts a list of SMILES strings
reward_fun = reward_funs[obj_num]
runner_factory = lambda: PolicyGradientRunner(grammar,
                              BATCH_SIZE=1,
                              reward_fun=reward_fun,
                              max_steps=60,
                              num_batches=2,
                              lr=0.02,
                              entropy_wgt=0.1,
                              # lr_schedule=shifted_cosine_schedule,
                              root_name='test',
                              preload_file_root_name=None,
                              plot_metrics=True,
                              save_location=test_data_dir,
                              metric_smooth=0.0,
                              decoder_type='graph_conditional',  # 'rnn_graph',# 'attention',
                              on_policy_loss_type='advantage_record',
                              rule_temperature_schedule=None, #lambda x: toothy_exp_schedule(x, scale=num_batches),
                              eps=0.0,
                              priors='conditional',
                              )

class Environment(TestCase):
    def test_populate_data_cache(self):
        out = populate_data_cache(test_data_dir)
        assert out == {'whatever2': 0.2857142984867096, 'whatever': 0.3173076808452606}

    def test_pick_model_to_run_empty_cache(self):
        model_factory = lambda: 'New Model'


        out = pick_model_to_run({}, model_factory, Dummy)
        assert out == 'New Model'

    def test_pick_model_to_run_full_cache(self):
        data_cache = {str(i): i for i in range(20)}
        model_factory = lambda: 'New Model'

        out = pick_model_to_run(data_cache, model_factory, Dummy)
        assert type(out) == str
        assert 10 <= int(out) <= 19

    def test_pick_model_to_run_small_cache(self):
        data_cache = {str(i): i for i in range(5)}
        model_factory = lambda: 'New Model'

        out = pick_model_to_run(data_cache, model_factory, Dummy)
        assert type(out) == str
        assert 0 <= int(out) <= 5

    def test_crossover(self):
        model1 = runner_factory()
        model2 = runner_factory()
        out = crossover(model1, model2)
        assert isinstance(out, PolicyGradientRunner)

    def test_mutate(self):
        model1 = runner_factory()
        out = mutate(model1)
        assert isinstance(out, PolicyGradientRunner)

    def test_classic_crossover(self):
        model1 = runner_factory()
        model2 = runner_factory()
        out = classic_crossover(model1, model2)
        assert isinstance(out, PolicyGradientRunner)

    def test_classic_mutate(self):
        model1 = runner_factory()
        out = classic_mutate(model1, delta_scale=0.1)
        assert isinstance(out, PolicyGradientRunner)

    def test_generate_root_name_1(self):
        old_root = 'foo'
        new_name = generate_root_name(old_root,{'fooa':0})
        assert new_name[:5] == 'foob#'
        assert len(new_name) > 5

    def test_generate_root_name_2(self):
        old_root = 'foo#' + str(uuid4())
        new_name = generate_root_name(old_root,{})
        assert new_name[:5] == 'fooa#'
        assert len(new_name) == len(old_root) +1
