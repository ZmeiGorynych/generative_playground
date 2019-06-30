from torch.utils.data import DataLoader

from generative_playground.models.decoder.decoder import get_node_decoder
from generative_playground.models.problem.rl.deepq import QLearningDataset, DistributionaDeepQModelWrapper, \
    collate_experiences, DistributionalDeepQWassersteinLoss
from generative_playground.utils.testing_utils import make_grammar


def test_deepq_distr_update_flow():
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