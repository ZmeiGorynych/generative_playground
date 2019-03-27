import logging
import random
import numpy as np
from unittest import TestCase
from generative_playground.codec.hypergraph import to_mol, HyperGraph
from generative_playground.codec.hypergraph_parser import hypergraph_parser, graph_from_graph_tree
from generative_playground.codec.hypergraph_grammar import evaluate_rules, HypergraphGrammar, HypergraphMaskGenerator
from rdkit.Chem import MolFromSmiles, AddHs, MolToSmiles, RemoveHs, Kekulize, BondType

smiles = ['CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1',
          'C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1',
          'N#Cc1ccc(-c2ccc(O[C@@H](C(=O)N3CCCC3)c3ccccc3)cc2)cc1']
smiles1 = smiles[0]


class TestStart(TestCase):
    def test_hypergraph_roundtrip(self):
        mol = MolFromSmiles(smiles1)
        hg = HyperGraph.from_mol(mol)
        re_mol = to_mol(hg)
        re_smiles = MolToSmiles(re_mol)
        assert re_smiles == smiles1

    def test_hypergraph_via_nx_graph_roundtrip(self):
        mol = MolFromSmiles(smiles1)
        hg = HyperGraph.from_mol(mol)
        re_mol = to_mol(hg.to_nx())
        re_smiles = MolToSmiles(re_mol)
        assert re_smiles == smiles1

    def test_parser_roundtrip(self):
        mol = MolFromSmiles(smiles1)
        tree = hypergraph_parser(mol)
        graph4 = evaluate_rules(tree.rules())
        mol4 = to_mol(graph4)
        smiles4 = MolToSmiles(mol4)
        assert smiles4 == smiles1

    def test_parser_roundtrip_no_rule_sequence(self):
        mol = MolFromSmiles(smiles1)
        tree = hypergraph_parser(mol)
        graph4 = graph_from_graph_tree(tree)
        mol4 = to_mol(graph4)
        smiles4 = MolToSmiles(mol4)
        assert smiles4 == smiles1

    def test_parser_roundtrip_via_indices(self):
        # TODO: cheating a bit here, reconstruction does fail for some smiles
        # in particular, rearranging subtrees to allow for isomorphism sometimes screws up chirality
        g = HypergraphGrammar()
        actions = g.strings_to_actions(smiles)
        re_smiles = g.decode_from_actions(actions)
        for old, new in zip(smiles, re_smiles):
            assert old == new

    def test_mask_gen(self):
        g = HypergraphGrammar()
        g.strings_to_actions(smiles) # that initializes g with the rules from these molecules
        g.calc_terminal_distance()
        batch_size = 10
        max_rules = 50
        all_actions = []
        next_action = [None for _ in range(batch_size)]
        mask_gen = HypergraphMaskGenerator(max_rules, g)
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
        # the test is that we get that far, producing valid molecules
        all_smiles = g.decode_from_actions(all_actions)
        for smile in all_smiles:
            print(smile)