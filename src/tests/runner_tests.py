from unittest import TestCase
from generative_playground.codec.hypergraph_grammar import HypergraphGrammar
from generative_playground.molecules.models.conditional_probability_model import CondtionalProbabilityModel
from generative_playground.models.pg_runner import PolicyGradientRunner
class TestStart(TestCase):
    def test_get_set_params_as_vector(self):
        grammar_cache = 'hyper_grammar_guac_10k_with_clique_collapse.pickle'  # 'hyper_grammar.pickle'
        first_runner = PolicyGradientRunner('hypergraph:' + grammar_cache,
                                            BATCH_SIZE=10,
                                            reward_fun=lambda x: 0,
                                            max_steps=60,
                                            num_batches=2,
                                            lr=0.05,
                                            entropy_wgt=0.0,
                                            # lr_schedule=shifted_cosine_schedule,
                                            root_name='test',
                                            preload_file_root_name=None,
                                            plot_metrics=True,
                                            save_location='./data',
                                            metric_smooth=0.0,
                                            decoder_type='graph_conditional',  # 'rnn_graph',# 'attention',
                                            on_policy_loss_type='advantage_record',
                                            rule_temperature_schedule=None,
                                            # lambda x: toothy_exp_schedule(x, scale=num_batches),
                                            eps=0.0,
                                            priors='conditional',
                                            )

        coeffs = first_runner.get_model_coeff_vector()
        coeffs[0] = 1
        first_runner.set_model_coeff_vector(coeffs)
        coeffs2 = first_runner.get_model_coeff_vector()
        assert coeffs2[0] == coeffs[0]
