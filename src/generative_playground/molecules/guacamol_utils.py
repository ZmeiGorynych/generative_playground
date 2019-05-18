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

from guacamol.scoring_function import ScoringFunction

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
    def __init__(self, cache_file, maximize_reward=False):
        # self.molecules = read(chembl_training_file) # TODO: the generator is allowed to see the molecule training file, right?
        with gzip.open(cache_file, 'rb') as f:
            self.data = pickle.load(f)
            # unique_
            # for s, r in self.data:
            if maximize_reward:
                self.data = sorted(list(set(self.data)), key=lambda x: x[1], reverse=True)
    def generate(self, number_samples: int): # the benchmarks use 10K samples
        return [x[0] for x in self.data[:number_samples]]

class MyGoalDirectedGenerator(GoalDirectedGenerator):
    def __init__(self, version):
        self.version = version
        self.num_benchmarks = len(goal_directed_benchmark_suite(self.version))
        self.obj_num = 0
        #TODO: want to generate the grammar from ChEMBL rather than ZINC
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
        cache_file = 'canned_' + self.version + '_' + str(self.obj_num) + 'do_0.5_lr4e-5_smiles.zip'
        root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        full_cache_file = os.path.realpath(root_location + '/train/pretrained/' + cache_file)
        #TODO: put the directory for the cache file here
        gen = DummyMoleculeGenerator(full_cache_file, maximize_reward=True)
        self.obj_num += 1
        if self.obj_num == self.num_benchmarks:
            self.obj_num == 0
        return gen.generate(number_molecules)