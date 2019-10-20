from unittest import TestCase
from generative_playground.codec.hypergraph_grammar import HypergraphGrammar
from generative_playground.molecules.models.conditional_probability_model import CondtionalProbabilityModel

class TestStart(TestCase):
    def test_get_set_params_as_vector(self):
        grammar_cache = 'hyper_grammar_guac_10k_with_clique_collapse.pickle'  # 'hyper_grammar.pickle'
        g = HypergraphGrammar.load(grammar_cache)
        m = CondtionalProbabilityModel(g)
        out = m.get_params_as_vector()

        out[0] = 1

        m.set_params_from_vector(out)
        out2 = m.get_params_as_vector()

        assert out2[0] == out[0]
