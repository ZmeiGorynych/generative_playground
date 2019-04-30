import logging
import random
import numpy as np
import os
from unittest import TestCase
from generative_playground.codec.hypergraph import to_mol, HyperGraph
from generative_playground.codec.hypergraph_parser import hypergraph_parser, graph_from_graph_tree
from generative_playground.codec.hypergraph_grammar import evaluate_rules, HypergraphGrammar, HypergraphMaskGenerator
from rdkit.Chem import MolFromSmiles, AddHs, MolToSmiles, RemoveHs, Kekulize, BondType
from generative_playground.codec.grammar_helper import grammar_eq, grammar_zinc, grammar_zinc_new
from generative_playground.codec.grammar_mask_gen import GrammarMaskGenerator
from generative_playground.codec.mask_gen_new_2 import GrammarMaskGeneratorNew
from generative_playground.codec.grammar_codec import CFGrammarCodec, zinc_tokenizer, zinc_tokenizer_new, eq_tokenizer
from generative_playground.codec.codec import get_codec

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