from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from generative_playground.molecules.guacamol_utils import MyGoalDirectedGenerator

my_gen = MyGoalDirectedGenerator('trivial')
assess_goal_directed_generation(goal_directed_molecule_generator=my_gen,
                                benchmark_version='trivial')