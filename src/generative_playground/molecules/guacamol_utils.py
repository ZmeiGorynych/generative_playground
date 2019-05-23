from guacamol.benchmark_suites import goal_directed_benchmark_suite
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.goal_directed_generator import GoalDirectedGenerator
import gzip, pickle
from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from abc import ABCMeta, abstractmethod
from typing import List, Optional
import numpy as np
import os, inspect
import glob
import random
from guacamol.scoring_function import ScoringFunction

root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def full_cache_path(file):
    return os.path.realpath(root_location + '/train/pretrained/' + file)

version_name_list = ['v1','v2', 'trivial']

class GuacamolGoalWrapper:
    def __init__(self, benchmark):
        self.benchmark = benchmark

    def __call__(self, smiles_list):
        out = [self.benchmark.objective.score(s) for s in smiles_list]
        return out
    @property
    def name(self):
        return self.benchmark.name

def guacamol_goal_scoring_functions(version_name):
    assert version_name in version_name_list, "Version name must be in " + str(version_name_list)
    benchmarks = goal_directed_benchmark_suite(version_name=version_name)
    out = [GuacamolGoalWrapper(b) for b in benchmarks]
    return out

class DummyMoleculeGenerator(DistributionMatchingGenerator):
    def __init__(self, cache_files,
                 maximize_reward=False,
                 keep_last=None,
                 make_unique=False):
        '''

        :param cache_files: a filename or list of filenames, that must live in the pretrained directory
        :param maximize_reward: whether to return the molecules with the highest reward
        '''
        self.maximize_reward = maximize_reward
        self.cache_files = cache_files

        if type(cache_files) == str:
            cache_files = [cache_files]


        pre_data = []
        for cache_file in cache_files:
            if not os.path.isfile(cache_file):
                full_cache_file = full_cache_path(cache_file)
            else:
                full_cache_file = cache_file
            print(full_cache_file)
            with gzip.open(full_cache_file, 'rb') as f:
                this_data = pickle.load(f)
                pre_data += this_data

        if keep_last is not None:
            pre_data = pre_data[-int(keep_last):]

        if make_unique:
            tmp_dict = {}
            for s, r in pre_data:
                if s not in tmp_dict.keys():
                    tmp_dict[s] = r
                else:
                    tmp_dict = max(r, tmp_dict[s])

            pre_data = list(tmp_dict.items())

        if maximize_reward:
            pre_data = sorted(pre_data, key=lambda x: x[1], reverse=True)

        self.data = pre_data

    def generate(self, number_samples: int): # the benchmarks use 10K samples
        if self.maximize_reward:
            return [x[0] for x in self.data[:number_samples]]
        else:
            return [x[0] for x in random.sample(self.data, number_samples)]


class MyGoalDirectedGenerator(GoalDirectedGenerator):
    def __init__(self, version):
        self.version = version
        self.num_benchmarks = len(goal_directed_benchmark_suite(self.version))
        self.obj_num = 0

    def generate_optimized_molecules(self,
                                     scoring_function: ScoringFunction,
                                     number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:
        """
        Given an objective function, generate molecules that score as high as possible.

        Args:
            scoring_function: scoring function
            number_molecules: number of molecules to generate
            starting_population: molecules to start the optimization from (optional)

        Returns:
            A list of SMILES strings for the generated molecules.
        """
        benchmarks = goal_directed_benchmark_suite(self.version)
        cache_files = ['guac_v2_{}do_0.5_lr4e-5_mark'.format(idx) for idx in range(20)]
        #TODO: put the directory for the cache file here
        gen = DummyMoleculeGenerator([cache_files[19]], maximize_reward=True)
        self.obj_num += 1
        if self.obj_num == self.num_benchmarks:
            self.obj_num == 0
        return gen.generate(number_molecules)
