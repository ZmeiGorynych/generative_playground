import logging
import random
import numpy as np
from unittest import TestCase
from rdkit.Chem import MolFromSmiles, AddHs, MolToSmiles, RemoveHs, Kekulize, BondType
from generative_playground.molecules.model_settings import get_settings
from generative_playground.codec.codec import get_codec
from generative_playground.codec.hypergraph_grammar import HypergraphGrammar

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
    def codec_test(self,
                   input,
                   molecules,
                   grammar):
        settings = get_settings(molecules, grammar)
        codec = get_codec(molecules,
                          grammar,
                          settings['max_seq_length'])
        actions = codec.strings_to_actions([input])
        re_input = codec.actions_to_strings(actions)
        self.assertEqual(input, re_input[0])

    def test_char_eq_codec(self):
        molecules = False
        grammar = False
        input = 'exp(1+sin(1*(x+2)))'
        self.codec_test(input, molecules, grammar)

    def test_grammar_eq_codec(self):
        molecules = False
        grammar = 'classic'
        input = 'exp(1+sin(1*(x+2)))'
        self.codec_test(input, molecules, grammar)

    def test_classic_char_codec(self):
        molecules = True
        grammar = False
        input = smiles1
        self.codec_test(input, molecules, grammar)

    def test_classic_grammar_codec(self):
        molecules = True
        grammar = 'classic'
        input = smiles1
        self.codec_test(input, molecules, grammar)

    def test_new_grammar_codec(self):
        molecules = True
        grammar = 'new'
        # not all valid SMILES can be represented by the new grammar, here is one string produced by it
        input = 'c1c(C([C@](=[C@](C[C@@]#[NH+])S[C@H]=[C@@H]2)N2N=[C@@H]2)=C2)ncc2c1cncn2'
        self.codec_test(input, molecules, grammar)

    def test_hypergraph_grammar_codec(self):
        molecules = True
        input = smiles1
        grammar_cache = 'tmp.pickle'
        grammar = 'hypergraph:' + grammar_cache
        # create a grammar cache inferred from our sample molecules
        g = HypergraphGrammar(cache_file=grammar_cache)
        g.strings_to_actions(smiles)
        self.codec_test(input, molecules, grammar)

