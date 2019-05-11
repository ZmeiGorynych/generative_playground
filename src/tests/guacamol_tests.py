from generative_playground.molecules.guacamol_utils import guacamol_goal_scoring_functions, version_name_list
from unittest import TestCase
import numpy as np

class TestGuacamolOjectives(TestCase):
    def test_scoring_functions(self):
        smiles = ['O', 'cccccc']
        for ver in version_name_list:
            print(ver)
            objectives = guacamol_goal_scoring_functions(ver)
            for obj in objectives:
                out = obj(smiles)
                print(obj.name, out)
                assert not np.isnan(sum(out)), "Objective returned NaN value!"
