from unittest import TestCase

from torch.utils.data import DataLoader
from generative_playground.models.problem.rl.deepq import QLearningDataset, DeepQModelWrapper, DeepQLoss, collate_experiences
from generative_playground.utils.testing_utils import make_grammar, make_decoder


class TestDeepQ(TestCase):
    def test_deepq_update_flow(self):
        # let's make some data
        grammar = make_grammar()
        decoder = make_decoder(grammar, output_spec={'action': len(grammar)})
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