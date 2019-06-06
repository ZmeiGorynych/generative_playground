import random
import numpy as np
import os
from unittest import TestCase
from generative_playground.codec.hypergraph_grammar import HypergraphGrammar, HypergraphMaskGenerator
from rdkit.Chem import MolFromSmiles
from generative_playground.codec.codec import get_codec
from generative_playground.codec.hypergraph_grammar import GrammarInitializer
from generative_playground.models.problem.policy import SoftmaxRandomSamplePolicy
from generative_playground.utils.gpu_utils import device
import torch

smiles = ['CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1',
          'C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1',
          'N#Cc1ccc(-c2ccc(O[C@@H](C(=O)N3CCCC3)c3ccccc3)cc2)cc1']
smiles1 = smiles[0]
max_seq_length = 50
batch_size = 10

def run_random_gen(mask_gen):
    all_actions = []
    next_action = [None for _ in range(batch_size)]

    while True:
        try:
            next_masks = mask_gen(next_action)
            next_action = []
            for mask in next_masks:
                inds = np.nonzero(mask)[0]
                next_act = random.choice(inds)
                next_action.append(next_act)
            all_actions.append(next_action)
        except StopIteration:
            break
    all_actions = np.array(all_actions).T
    return all_actions



class TestStart(TestCase):
    def test_classic_mask_gen_equations(self):
        molecules = False
        grammar = 'classic'
        codec = get_codec(molecules, grammar, max_seq_length)
        actions = run_random_gen(codec.mask_gen)
        all_eqs = codec.actions_to_strings(actions)
        # the only way of testing correctness we have here is whether the equations parse correctly
        parsed_eqs = codec.strings_to_actions(all_eqs)

    def test_classic_mask_gen_molecules(self):
        molecules = True
        grammar = 'classic'
        codec = get_codec(molecules, grammar, max_seq_length)
        actions = run_random_gen(codec.mask_gen)
        new_smiles = codec.actions_to_strings(actions)
        # the SMILES produced by that grammar are NOT guaranteed to be valid,
        # so can only check that the decoding completes without errors and is grammatically valid
        parsed_smiles = codec.strings_to_actions(new_smiles)

    def test_custom_grammar_mask_gen(self):
        molecules = True
        grammar = 'new'
        codec = get_codec(molecules, grammar, max_seq_length)
        self.generate_and_validate(codec)

    def test_hypergraph_mask_gen(self):
        molecules = True
        grammar_cache = 'tmp.pickle'
        grammar = 'hypergraph:' + grammar_cache
        # create a grammar cache inferred from our sample molecules
        g = HypergraphGrammar(cache_file=grammar_cache)
        if os.path.isfile(g.cache_file):
            os.remove(g.cache_file)
        g.strings_to_actions(smiles)
        codec = get_codec(molecules, grammar, max_seq_length)
        self.generate_and_validate(codec)

    def generate_and_validate(self, codec):
        actions = run_random_gen(codec.mask_gen)
        # the test is that we get that far, producing valid molecules
        all_smiles = codec.actions_to_strings(actions)
        for smile in all_smiles:
            self.assertIsNot(MolFromSmiles(smile), None)


# TODO: all these tests refer to an obsolete version of the model, where the first nonterminal was always expanded
# should purge all these tests as well as the model itself
    # def test_hypergraph_mask_gen_conditional_priors(self):
    #     tmp_file = 'tmp2.pickle'
    #     gi = GrammarInitializer(tmp_file)
    #     gi.delete_cache()
    #     # now create a clean new one
    #     gi = GrammarInitializer(tmp_file)
    #     # run a first run for 10 molecules
    #     gi.init_grammar(20)
    #     gi.grammar.check_attributes()
    #
    #     mask_gen = HypergraphMaskGenerator(30, gi.grammar, priors='conditional')
    #
    #     all_actions = []
    #     next_action = [None for _ in range(2)]
    #     policy = SoftmaxRandomSamplePolicy()
    #     while True:
    #         try:
    #             mask_gen.apply_action(next_action)
    #             cond_priors = mask_gen.action_prior_logits()
    #             cond_priors_pytorch = torch.from_numpy(cond_priors).to(device=device, dtype=torch.float32)
    #             next_action = policy(cond_priors_pytorch).cpu().detach().numpy()
    #             all_actions.append(next_action)
    #         except StopIteration:
    #             break
    #     all_actions = np.array(all_actions).T
    #     all_smiles = gi.grammar.actions_to_strings(all_actions)
    #     for smile in all_smiles:
    #         self.assertIsNot(MolFromSmiles(smile), None)

    def test_hypergraph_mask_gen_unconditional_priors(self):
        tmp_file = 'tmp2.pickle'
        gi = GrammarInitializer(tmp_file)
        gi.delete_cache()
        # now create a clean new one
        gi = GrammarInitializer(tmp_file)
        # run a first run for 10 molecules
        gi.init_grammar(20)
        gi.grammar.check_attributes()

        mask_gen = HypergraphMaskGenerator(30, gi.grammar, priors=True)

        all_actions = []
        next_action = [None for _ in range(2)]
        policy = SoftmaxRandomSamplePolicy()
        while True:
            try:
                mask_gen.apply_action(next_action)
                cond_priors = mask_gen.action_prior_logits()
                cond_priors_pytorch = torch.from_numpy(cond_priors).to(device=device, dtype=torch.float32)
                next_action = policy(cond_priors_pytorch).cpu().detach().numpy()
                all_actions.append(next_action)
            except StopIteration:
                break
        all_actions = np.array(all_actions).T
        all_smiles = gi.grammar.actions_to_strings(all_actions)
        for smile in all_smiles:
            self.assertIsNot(MolFromSmiles(smile), None)

    def test_hypergraph_mask_gen_no_priors(self):
        tmp_file = 'tmp2.pickle'
        gi = GrammarInitializer(tmp_file)
        gi.delete_cache()
        # now create a clean new one
        gi = GrammarInitializer(tmp_file)
        # run a first run for 10 molecules
        gi.init_grammar(20)
        gi.grammar.check_attributes()

        mask_gen = HypergraphMaskGenerator(30, gi.grammar, priors=False)

        all_actions = []
        next_action = [None for _ in range(2)]
        policy = SoftmaxRandomSamplePolicy()
        while True:
            try:
                mask_gen.apply_action(next_action)
                cond_priors = mask_gen.action_prior_logits()
                cond_priors_pytorch = torch.from_numpy(cond_priors).to(device=device, dtype=torch.float32)
                next_action = policy(cond_priors_pytorch).cpu().detach().numpy()
                all_actions.append(next_action)
            except StopIteration:
                break
        all_actions = np.array(all_actions).T
        all_smiles = gi.grammar.actions_to_strings(all_actions)
        for smile in all_smiles:
            self.assertIsNot(MolFromSmiles(smile), None)

    def test_hypergraph_mask_gen_step(self):
        tmp_file = 'tmp2.pickle'
        gi = GrammarInitializer(tmp_file)
        gi.delete_cache()
        # now create a clean new one
        gi = GrammarInitializer(tmp_file)
        # run a first run for 10 molecules
        gi.init_grammar(20)
        gi.grammar.check_attributes()
        mask_gen = HypergraphMaskGenerator(30, gi.grammar, priors=True)
        batch_size = 2
        next_action = (None, [None for _ in range(batch_size)])
        while True:
            try:
                graphs, node_mask, full_logit_priors = mask_gen.step(next_action)
                next_node = np.argmax(node_mask, axis=1)
                next_action_ = [np.argmax(full_logit_priors[b, next_node[b]]) for b in range(batch_size)]
                next_action = (next_node, next_action_)
            except StopIteration:
                break
