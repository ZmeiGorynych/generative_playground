from collections import OrderedDict
from generative_playground.molecules.lean_settings import molecules_root_location
from generative_playground.codec.parent_codec import GenericCodec
from generative_playground.codec.hypergraph import (
    HyperGraph, HypergraphTree, replace_nonterminal, to_mol, MolToSmiles, MolFromSmiles, hypergraphs_are_equivalent,
    put_parent_node_first)
from generative_playground.codec.hypergraph_parser import hypergraph_parser, tree_with_rule_inds_to_list_of_tuples
from generative_playground.molecules.data_utils.zinc_utils import get_smiles_from_database
import pickle
import os, copy
import numpy as np
import math
from functools import lru_cache

grammar_data_location = molecules_root_location + 'data/grammar/'


def full_location(filename):
    return os.path.realpath(grammar_data_location + filename)


class HypergraphGrammar(GenericCodec):
    def __init__(self, cache_file='tmp.pickle', max_len=None, isomorphy=False):
        self.id_by_parent = {'DONE': [0]} # from str(parent_node) to rule index
        self.parent_by_id = {0: 'DONE'} # from rule index to str(parent_node)
        self.rules = [None]# list of HyperGraphFragments
        self.rule_frequency_dict = {}
        self.node_data_index = OrderedDict()
        self.rate_tracker = []
        self.candidate_counter = 0
        self.cache_file = full_location(cache_file)
        self.terminal_distance_by_parent = {}
        self._rule_term_dist_deltas = []
        self.shortest_rule_by_parent = {}
        self.last_tree_processed = None
        self.MAX_LEN = max_len # only used to pad string_to_actions output, factor out?
        self.PAD_INDEX = 0
        self.conditional_frequencies = {}
        # self.isomorphy_match = isomorphy

    def __len__(self):
        return len(self.rules)

    def check_attributes(self):
        for rule in self.rules:
            if rule is not None:
                for this_node in rule.node.values():
                    assert hasattr(this_node, 'rule_id')
                    assert hasattr(this_node, 'node_index')

    def feature_len(self):
        return len(self)

    def delete_cache(self):
        if os.path.isfile(self.cache_file):
            os.remove(self.cache_file)

    @property
    def grammar(self):
        return self

    @property
    def rule_term_dist_deltas(self): # write-protect that one
        return self._rule_term_dist_deltas

    @classmethod
    def load(Class, filename):
        with open(full_location(filename), 'rb') as f:
            self = pickle.load(f)
        return self


    def get_log_frequencies(self):
        out = np.zeros(len(self.rules))
        for ind, value in self.rule_frequency_dict.items():
            out[ind] = math.log(value)
        return out

    @lru_cache()
    def get_conditional_log_frequencies_single_query(self, x, default=-3):
        out = default*np.ones(len(self.rules))
        if x in self.conditional_frequencies:
            for ind, value in self.conditional_frequencies[x].items():
                out[ind] = math.log(value)
        return out

    def decode_from_actions(self, actions):
        '''

        :param actions: batch_size x max_out_len longs
        :return: batch_size of decoded SMILES strings
        '''
        # just loop through the batch
        out = []
        for action_seq in actions:
            rules = [self.rules[i] for i in action_seq if not self.is_padding(i)]
            graph = evaluate_rules(rules)
            mol = to_mol(graph)
            smiles = MolToSmiles(mol)
            out.append(smiles)
        return out

    def _smiles_to_tree_gen(self, smiles):
        assert type(smiles) == list or type(smiles) == tuple, "Input must be a list or a tuple"
        for smile in smiles:
            mol = MolFromSmiles(smile)
            assert mol is not None, "SMILES String could not be parsed: " + smile
            try:
                tree = hypergraph_parser(mol)
            except Exception as e:
                print(str(e))
                continue
            yield self.normalize_tree(tree)

    def raw_strings_to_actions(self, smiles):
        '''
        Convert a list of valid SMILES string to actions
        :param smiles: a list of valid SMILES strings
        :return:
        '''
        actions = []
        for norm_tree in self._smiles_to_tree_gen(smiles):
            these_actions = [rule.rule_id for rule in norm_tree.rules()]
            actions.append(these_actions)
        return actions

    def strings_to_actions(self, smiles, MAX_LEN=100):
        list_of_action_lists = self.raw_strings_to_actions(smiles)
        actions = [a + [self.PAD_INDEX] * (MAX_LEN - len(a)) for a in list_of_action_lists ]
        return np.array(actions)

    def normalize_tree(self, tree):
        # as we're replacing the original hypergraph from the parser with an equivalent node from our rules list,
        # which could have a different order of nonterminals, need to reorder subtrees to match
        new_subtrees = [self.normalize_tree(subtree) for subtree in tree]
        child_id_to_subtree = {child_id: subtree for child_id, subtree in zip(tree.node.child_ids(), new_subtrees)}
        rule_id, node_id_map = self.rule_to_index(tree.node)
        reordered_subtrees = [child_id_to_subtree[node_id_map[child_id]] for child_id in self.rules[rule_id].child_ids()]
        new_tree = HypergraphTree(node=self.rules[rule_id], children=reordered_subtrees)
        new_tree.node.rule_id = rule_id
        self.last_tree_processed = new_tree
        return new_tree

    def calc_terminal_distance(self):
        self.terminal_distance_by_parent = {}
        self._rule_term_dist_deltas = []
        self.shortest_rule_by_parent = {}
        self.terminal_distance_by_parent = {parent_str: float('inf') for parent_str in self.id_by_parent.keys()}
        while True:
            prev_terminal_distance = copy.deepcopy(self.terminal_distance_by_parent)
            for rule in self.rules:
                if rule is None: # after we're done expanding, padding resolves to this
                    term_dist_candidate = 0
                    this_hash = 'DONE'
                else:
                    term_dist_candidate = 1 + sum([self.terminal_distance_by_parent[str(child)] for child in rule.children()])
                    this_hash = str(rule.parent_node())
                if self.terminal_distance_by_parent[this_hash] > term_dist_candidate:
                    self.terminal_distance_by_parent[this_hash] = term_dist_candidate
                    self.shortest_rule_by_parent[this_hash] = rule
            if self.terminal_distance_by_parent == prev_terminal_distance:
                break

        for r, rule in enumerate(self.rules):
            if rule is None:
                rule_term_dist_delta = float('-inf') # the padding rule
            else:
                rule_term_dist_delta = 1 + sum([self.terminal_distance_by_parent[str(child)] for child in rule.children()])\
                                   - self.terminal_distance_by_parent[str(rule.parent_node())]
                assert rule_term_dist_delta >= 0

            self._rule_term_dist_deltas.append(rule_term_dist_delta)

        self._rule_term_dist_deltas = np.array(self._rule_term_dist_deltas)

        assert min(self._rule_term_dist_deltas[1:]) >= 0
        assert len(self._rule_term_dist_deltas) == len(self.rules)

        print('terminal distance calculated!')

    def terminal_distance(self, graph):
        if graph is None:
            return 0
        else:
            return sum([self.terminal_distance_by_parent[str(child)] for child in graph.children()])

    def get_mask(self, next_rule_string, max_term_dist):
        out = np.zeros((len(self)))
        out[self.id_by_parent[next_rule_string]] = 1
        out[self.rule_term_dist_deltas > max_term_dist] = 0
        # for i, rule in enumerate(self.rules):
        #     if i in self.id_by_parent[next_rule_string] and self.rule_term_dist_deltas[i] <= max_term_dist:
        #         out.append(1)
        #     else:
        #         out.append(0)
        assert any(out), "Mask must allow at least one rule"
        return out


    def process_candidates(self, rules):
        for rule in rules:
            self.rule_to_index(rule)

    def rule_to_index(self, rule: HyperGraph, no_new_rules=False):
        self.candidate_counter +=1
        parent_node = rule.parent_node()

        if str(parent_node) not in self.id_by_parent:
            self.id_by_parent[str(parent_node)] = []

        # only check the equivalence against graphs with matching parent node
        for rule_id in self.id_by_parent[str(parent_node)]:
            mapping = hypergraphs_are_equivalent(self.rules[rule_id], rule)
            if mapping is not None:
                # if we found a match, we're done!
                return rule_id, mapping
        # if got this far, no match so this is a new rule
        if no_new_rules:
            raise ValueError("Unknown rule hypergraph " + str(rule))
        self.add_rule(rule)
        return (len(self.rules)-1), {i: i for i in rule.node.keys()}

    def add_rule(self, rule):
        rule = put_parent_node_first(rule)

        parent_node = rule.parent_node()
        # add more information to the rule nodes, to be used later
        for n, node in enumerate(rule.node.values()):
            node.rule_id = len(self.rules)
            node.node_index = n
        self.rules.append(rule)
        for node in rule.node.values():
            self.index_node_data(node)
        new_rule_index = len(self.rules)-1
        self.id_by_parent[str(parent_node)].append(new_rule_index)
        self.parent_by_id[new_rule_index] = str(parent_node)
        self.rate_tracker.append((self.candidate_counter, len(self.rules)))
        # print(self.rate_tracker[-1])
        if self.cache_file is not None:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self, f)


    def index_node_data(self, node):
        for fn in node.data.keys():
            if fn not in self.node_data_index:
                self.node_data_index[fn] = OrderedDict()
            if node.data[fn] not in self.node_data_index[fn]:
                self.node_data_index[fn][node.data[fn]] = len(self.node_data_index[fn])

    def node_data_index_length(self):
        # an extra slot needed for 'other' for each fieldname
        return len(self.node_data_index) + sum([len(x) for x in self.node_data_index.values()])

    def reset_rule_frequencies(self):
        self.conditional_frequencies = {}
        self.rule_frequency_dict = {}

    def count_rule_frequencies(self, trees):
        for tree in trees:
            these_tuples = tree_with_rule_inds_to_list_of_tuples(tree)
            for p, nt, c in these_tuples:
                if (p, nt) not in self.conditional_frequencies:
                    self.grammar.conditional_frequencies[(p, nt)] = {}
                if c not in self.conditional_frequencies[(p, nt)]:
                    self.conditional_frequencies[(p, nt)][c] = 1
                else:
                    self.conditional_frequencies[(p, nt)][c] += 1

            these_actions = [rule.rule_id for rule in tree.rules()]
            for a in these_actions:
                if a not in self.rule_frequency_dict:
                    self.rule_frequency_dict[a] = 0
                self.rule_frequency_dict[a] += 1


def check_full_equivalence(graph1, graph2):
    for node1, node2 in zip(graph1.node.values(), graph2.node.values()):
        assert str(node1) == str(node2)
        for edge_id_1, edge_id_2 in zip(node1.edge_ids, node2.edge_ids):
            assert graph1.edges[edge_id_1].type == graph2.edges[edge_id_2].type

def apply_rule(start_graph, rule, loc=None):
    if loc == None:
        child_ids = start_graph.child_ids()
        if len(child_ids):
            loc = child_ids[-1]
        else:
            raise ValueError("Start graph has no children!")
    start_graph = replace_nonterminal(start_graph, loc, rule)
    return start_graph


def evaluate_rules(rules):
    start_graph = rules[0].clone()
    for num, rule in enumerate(rules[1:]):
        if rule is not None:  # None are the padding rules
            start_graph = apply_rule(start_graph, rule)
    return start_graph

class GrammarInitializer:
    def __init__(self, filename, grammar_class=HypergraphGrammar):
        self.grammar_filename = self.full_grammar_filename(filename)
        self.own_filename = self.full_own_filename(filename)
        self.max_len = 0 # maximum observed number of rules so far
        self.last_processed = -1
        self.new_rules = []
        self.frequency_dict = {}
        self.total_len = 0
        self.stats = {}
        if os.path.isfile(self.grammar_filename):
            self.grammar = grammar_class.load(filename)
        else:
            self.grammar = grammar_class(cache_file=filename)

    def full_grammar_filename(self, filename):
        return grammar_data_location + filename

    def full_own_filename(self, filename):
        return grammar_data_location + 'init_' + filename

    def save(self):
        with open(self.own_filename, 'wb') as f:
            pickle.dump(self, f)
        with open(self.grammar_filename, 'wb') as f:
            pickle.dump(self.grammar, f)

    @classmethod
    def load(Class, filename):
        with open(filename, 'rb') as f:
            out = pickle.load(f)
        assert type(out) == Class
        return out

    def delete_cache(self):
        if os.path.isfile(self.own_filename):
            os.remove(self.own_filename)
        if os.path.isfile(self.grammar_filename):
            os.remove(self.grammar_filename)
        self.grammar.delete_cache()

    def init_grammar(self, max_num_mols):
        L = get_smiles_from_database(max_num_mols)
        for ind, smiles in enumerate(L):
            if ind >= max_num_mols:
                break
            if ind > self.last_processed: # don't repeat
                try:
                    # this causes g to remember all the rules occurring in these molecules
                    these_actions = self.grammar.raw_strings_to_actions([smiles])
                    this_tree = self.grammar.last_tree_processed
                    these_tuples = tree_with_rule_inds_to_list_of_tuples(this_tree)
                    for p, nt, c in these_tuples:
                        if (p,nt) not in self.grammar.conditional_frequencies:
                            self.grammar.conditional_frequencies[(p, nt)] = {}
                        if c not in self.grammar.conditional_frequencies[(p, nt)]:
                            self.grammar.conditional_frequencies[(p, nt)][c] = 1
                        else:
                            self.grammar.conditional_frequencies[(p, nt)][c] += 1
                    # count the frequency of the occurring rules
                    for aa in these_actions:
                        for a in aa:
                            if a not in self.grammar.rule_frequency_dict:
                                self.grammar.rule_frequency_dict[a] = 0
                            self.grammar.rule_frequency_dict[a] += 1

                    lengths = [len(x) for x in these_actions]
                    new_max_len = max(lengths)
                    self.total_len += sum(lengths)
                    if new_max_len > self.max_len:
                        self.max_len = new_max_len
                        print("Max len so far:", self.max_len)
                except Exception as e: #TODO: fix this, make errors not happen ;)
                    print(e)
                self.last_processed = ind
                # if we discovered a new rule, remember that
                if not len(self.new_rules) or self.grammar.rate_tracker[-1][-1] > self.new_rules[-1][-1]:
                    self.new_rules.append((ind,*self.grammar.rate_tracker[-1]))
                    print(self.new_rules[-1])
            if ind % 10 == 9:
                self.save()
            if ind % 100 == 0 and ind > 0:
                self.stats[ind] = {
                    'max_len': self.max_len,
                    'avg_len': self.total_len / ind,
                    'num_rules': len(self.grammar.rules),
                }
        self.grammar.calc_terminal_distance()
        return self.max_len # maximum observed molecule length
