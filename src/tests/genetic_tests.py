import random
import numpy as np
import os, sys, inspect
from unittest import TestCase
from generative_playground.models.problem.genetic.genetic_opt import populate_data_cache, pick_model_to_run

my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
test_root = os.path.realpath(my_location)
test_data_dir = os.path.realpath(test_root + '/test_data')

class Environment(TestCase):
    def test_populate_data_cache(self):
        out = populate_data_cache(test_data_dir)
        assert out == {'whatever2': 0.2857142984867096, 'whatever': 0.3173076808452606}

    def test_pick_model_to_run_empty_cache(self):
        model_factory = lambda: 'New Model'
        class Dummy:
            @classmethod
            def load(cls, x):
                return x

        out = pick_model_to_run({}, model_factory, Dummy)
        assert out == 'New Model'

    def test_pick_model_to_run_full_cache(self):
        data_cache = {str(i): i for i in range(20)}
        model_factory = lambda: 'New Model'

        class Dummy:
            @classmethod
            def load(cls, x):
                return x

        out = pick_model_to_run(data_cache, model_factory, Dummy)
        assert type(out) == str
        assert 10 <= int(out) <= 19

    def test_pick_model_to_run_small_cache(self):
        data_cache = {str(i): i for i in range(5)}
        model_factory = lambda: 'New Model'

        class Dummy:
            @classmethod
            def load(cls, x):
                return x

        out = pick_model_to_run(data_cache, model_factory, Dummy)
        assert type(out) == str
        assert 0 <= int(out) <= 5
