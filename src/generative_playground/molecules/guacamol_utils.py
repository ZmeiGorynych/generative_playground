from guacamol.benchmark_suites import goal_directed_benchmark_suite
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.assess_distribution_learning import assess_distribution_learning
from abc import ABCMeta, abstractmethod
from typing import List, Optional

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

class MyDistributionMatchingGenerator(DistributionMatchingGenerator):
    def __init__(self, chembl_training_file: str):
        # self.molecules = read(chembl_training_file) # TODO: the generator is allowed to see the molecule training file, right?
        pass
    def generate(self, number_samples: int): # the benchmarks use 10K samples
        pass

class MyGoalDirectedGenerator(GoalDirectedGenerator):
    def __init__(self, molecules_training_file):
        pass
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