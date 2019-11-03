import random
import numpy as np
import os, sys, inspect
from unittest import TestCase

from generative_playground.models.param_sampler import ParameterSampler, extract_params_rewards, values_to_percentiles

my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
test_root = os.path.realpath(my_location)
data_file = os.path.realpath(test_root + '/../generative_playground/molecules/train/genetic/data/geneticA9_graph.zip')



class SamplerTest(TestCase):
    def test_param_sampler(self):
        params, rewards = extract_params_rewards(data_file)
        sampler = ParameterSampler(params, rewards)
        sample = sampler.sample()
        print('done!')