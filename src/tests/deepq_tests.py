from unittest import TestCase

from torch.utils.data import DataLoader
from generative_playground.models.problem.rl.deepq import *
from generative_playground.utils.testing_utils import make_grammar, make_decoder
from generative_playground.models.decoder.decoder import get_node_decoder

class TestDeepQ(TestCase):
    def test_deepq_update_flow(self):
        # let's make some data
        # grammar = make_grammar()
        # decoder = make_decoder(grammar, output_spec={'action': len(grammar)})
        tmp_file = 'tmp2.pickle'
        make_grammar(tmp_file)
        grammar = 'hypergraph:' + tmp_file
        max_seq_length = 30
        decoder_type = 'attn_graph_node'
        decoder, stepper = get_node_decoder(grammar=grammar,
                                            max_seq_length=max_seq_length,
                                            decoder_type=decoder_type,
                                            rule_policy=None,
                                            batch_size=3)
        out = decoder()
        data = QLearningDataset()
        data.update_data(out)

        # Now let's load it and evaluate the policy model on it
        value_model = decoder.stepper.model
        deepq_wrap = DeepQModelWrapper(value_model)
        loader = DataLoader(dataset=data,
                            batch_size=3,
                            shuffle=True,
                            collate_fn=collate_experiences)

        for data_batch in loader:
            deepq_out = deepq_wrap(data_batch)
            break

        loss = DeepQLoss()
        this_loss = loss(deepq_out)
        this_loss.backward()
        print('done!')

    def test_deepq_distr_update_flow(self):
        # let's make some data
        tmp_file = 'tmp2.pickle'
        make_grammar(tmp_file)
        grammar = 'hypergraph:' + tmp_file
        max_seq_length = 30
        decoder_type = 'attn_graph_distr'
        decoder, stepper = get_node_decoder(grammar=grammar,
                                            max_seq_length=max_seq_length,
                                            decoder_type=decoder_type,
                                            rule_policy=None,
                                            batch_size=3)
        out = decoder()
        data = QLearningDataset()
        data.update_data(out)

        # Now let's load it and evaluate the policy model on it
        value_model = decoder.stepper.model
        deepq_wrap = DistributionaDeepQModelWrapper(value_model)
        loader = DataLoader(dataset=data,
                            batch_size=3,
                            shuffle=True,
                            collate_fn=collate_experiences)

        for data_batch in loader:
            deepq_out = deepq_wrap(data_batch)
            break

        loss = DistributionalDeepQWassersteinLoss()
        this_loss = loss(deepq_out)
        this_loss.backward()
        print('done!')