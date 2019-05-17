# from generative_playground.codec.hypergraph_grammar import HypergraphMaskGenerator
#
# from generative_playground.codec.hypergraph_parser import hypergraph_parser, \
#     check_validity
# from generative_playground.molecules.model_settings import get_settings
# from collections import OrderedDict
# from rdkit.Chem import MolFromSmiles, MolToSmiles
# import copy
# import random, os
# import numpy as np
#
#
#
#
# if __name__ == '__main__':
#     from generative_playground.codec.hypergraph_grammar import HypergraphGrammar, evaluate_rules, hypergraphs_are_equivalent
#
#     settings = get_settings(molecules=True, grammar='new')
#     thresh = 100000
#     # Read in the strings
#     f = open(settings['source_data'], 'r')
#     L = []
#     for line in f:
#         line = line.strip()
#         L.append(line)
#         if len(L) > thresh:
#             break
#     f.close()
#
#     fn = "rule_hypergraphs.pickle"
#     max_rules = 50
#     if os.path.isfile(fn):
#         rm = HypergraphGrammar.load(fn)
#     else:
#         rm = HypergraphGrammar(cache_file=fn)
#         bad_smiles = []
#         for num, smile in enumerate(L[:100]):
#             try:
#                 smile = MolToSmiles(MolFromSmiles(smile))
#                 print(smile)
#                 mol = MolFromSmiles(smile)
#                 actions = rm.strings_to_actions([smile])
#                 re_smile = rm.decode_from_actions(actions)[0]
#                 mol = MolFromSmiles(smile)
#                 if re_smile != smile:
#                     print("SMILES reconstruction wasn't perfect for " + smile)
#                 print(re_smile)
#
#             except Exception as e:
#                 print(e)
#                 bad_smiles.append(smile)
#
#     rm.calc_terminal_distance()
#
#     # now let's write a basic for-loop to create molecules
#     batch_size = 10
#     all_actions = []
#     next_action = [None for _ in range(batch_size)]
#     mask_gen = HypergraphMaskGenerator(max_rules, rm)
#     while True:
#         try:
#             next_masks = mask_gen(next_action)
#             next_action = []
#             for mask in next_masks:
#                 inds = np.nonzero(mask)[0]
#                 next_act = random.choice(inds)
#                 next_action.append(next_act)
#             all_actions.append(next_action)
#         except StopIteration:
#             break
#
#     all_actions = np.array(all_actions).T
#     all_smiles = rm.decode_from_actions(all_actions)
#     for smile in all_smiles:
#         print(smile)
#
#
#
#
#     # look at isomorphisms between them
#     def rule_has_terminals(x):
#         return any([node.is_terminal for node in x.node.values()])
#
#     def rule_has_nonterminals(graph):
#         return len(graph.child_ids()) >0
#     to_atom_rules = [x for x in rm.rules if rule_has_terminals(x)]
#     check_rules = [x for x in rm.rules if rule_has_terminals(x) and rule_has_nonterminals(x)]
#     print(len(to_atom_rules))
#     for i in range(20):
#         print(i,len([x for x in rm.rules if len(x.node)==i]))
#     print('aaa')
#
#
# [print(len(x.node)) for x in rm.rules]
#
# # TODO: move these to unit tests
# # graph3 = graph_from_graph_tree(copy.deepcopy(graph_tree))
# # mol3 = to_mol(graph3)
# # smiles3 = MolToSmiles(mol3)
# # print(smiles3)
# #
# # rules_list = graph_tree_to_rules_list(copy.deepcopy(graph_tree))
# # graph2 = apply_rules(copy.deepcopy(rules_list))
# # mol2 = to_mol(graph2)
# # smiles2 = MolToSmiles(mol2)
# # print(smiles2)