from collections import OrderedDict
from generative_playground.molecules.lean_settings import get_data_location
import frozendict
from generative_playground.codec.parent_codec import GenericCodec
from generative_playground.codec.hypergraph import HyperGraphFragment, HypergraphTree, replace_nonterminal, to_mol, MolToSmiles, MolFromSmiles
from generative_playground.codec.hypergraph_parser import hypergraph_parser
import networkx as nx
import pickle
import zipfile
import os, copy
import numpy as np
from pathlib import Path


class HypergraphGrammar(GenericCodec):
    def __init__(self, cache_file=None, max_len=None):
        self.id_by_parent = {'DONE': [0]} # from str(parent_node) to rule index
        self.parent_by_id = {0: 'DONE'} # from rule index to str(parent_node)
        self.rules = [None] # list of HyperGraphFragments
        self.rate_tracker = []
        self.candidate_counter = 0
        self.cache_file = cache_file
        self.terminal_distance_by_parent = {}
        self.rule_term_dist_deltas = []
        self.shortest_rule_by_parent = {}
        self.MAX_LEN = max_len # only used to pad string_to_actions output, factor out?
        self.PAD_INDEX = 0

    def __len__(self):
        return len(self.rules)

    def feature_len(self):
        return len(self)

    @classmethod
    def load(Class, filename):
        with open(filename, 'rb') as f:
            self = pickle.load(f)
        return self

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

    def raw_strings_to_actions(self, smiles):
        '''
        Convert a list of valid SMILES string to actions
        :param smiles: a list of valid SMILES strings
        :return:
        '''
        assert type(smiles) == list or type(smiles) == tuple, "Input must be a list or a tuple"
        actions = []
        for smile in smiles:
            these_actions = []
            mol = MolFromSmiles(smile)
            assert mol is not None, "SMILES String could not be parsed: " + smile
            tree = hypergraph_parser(mol)
            norm_tree = self.normalize_tree(tree)
            these_actions = [rule.rule_id for rule in norm_tree.rules()]
            actions.append(these_actions)
        return actions

    def strings_to_actions(self, list_of_action_lists, MAX_LEN=100):
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
        return new_tree

    def calc_terminal_distance(self):
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

        for rule in self.rules:
            if rule is None:
                rule_term_dist_delta = 0
            else:
                rule_term_dist_delta = 1 + sum([self.terminal_distance_by_parent[str(child)] for child in rule.children()])\
                                   - self.terminal_distance_by_parent[str(rule.parent_node())]
            self.rule_term_dist_deltas.append(rule_term_dist_delta)

        print('terminal distance calculated!')

    def terminal_distance(self, graph):
        if graph is None:
            return 0
        else:
            return sum([self.terminal_distance_by_parent[str(child)] for child in graph.children()])

    def get_mask(self, next_rule_string, max_term_dist):
        out = []
        for i, rule in enumerate(self.rules):
            if i in self.id_by_parent[next_rule_string] and self.rule_term_dist_deltas[i] <= max_term_dist:
                out.append(1)
            else:
                out.append(0)
        assert any(out), "Mask must allow at least one rule"
        return out


    def process_candidates(self, rules):
        for rule in rules:
            self.rule_to_index(rule)

    def rule_to_index(self, rule: HyperGraphFragment, no_new_rules=False):
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
        rule.is_rule = True
        self.rules.append(rule)
        new_rule_index = len(self.rules)-1
        self.id_by_parent[str(parent_node)].append(new_rule_index)
        self.parent_by_id[new_rule_index] = str(parent_node)
        self.rate_tracker.append((self.candidate_counter, len(self.rules)))
        # print(self.rate_tracker[-1])
        if self.cache_file is not None:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self, f)

        return new_rule_index, {i: i for i in rule.node.keys()}# {i:i for i in rule.edges.keys()}

class HypergraphMaskGenerator:
    def __init__(self, max_len, grammar):
        self.grammar = grammar
        self.MAX_LEN = max_len
        self.graphs = None
        self.t = 0

    def reset(self):
        self.graphs = None
        self.t = 0

    def __call__(self, last_action):
        '''
        Consumes one action at a time, responds with the mask for next action
        : param last_action: previous action, array of ints of len = batch_size; None for the very first step
        '''
        if self.t >= self.MAX_LEN:
            raise StopIteration("maximum sequence length exceeded for decoder")

        # apply the last action
        if last_action[0] is None:
            # first call
            self.graphs = [None for _ in range(len(last_action))]
        else:
            # evaluate the rule; assume the rule is valid
            for ind, graph, last_act in zip(range(len(self.graphs)),self.graphs, last_action):
                last_rule = self.grammar.rules[last_act]
                if graph is None:
                    # first call, graph is being created
                    assert last_rule.parent_node() is None
                    graph = last_rule.clone()
                else:
                    nonterminals_left = graph.nonterminal_ids()
                    if len(nonterminals_left):
                        # choose the next node to expand
                        expand_location = nonterminals_left[-1]
                        graph = replace_nonterminal(graph,
                                                expand_location,
                                                last_rule)
                    else:
                        # no more nonterminals, nothing for us to do but assert rule validity
                        assert last_rule == None
        # validity check
                assert graph.parent_node_id is None
                self.graphs[ind] = graph

        # now go over the graphs and determine the next mask
        masks = []
        for graph in self.graphs:
            if graph is None:
                next_rule_string = 'None'
            else:
                nonterminals_left = graph.nonterminal_ids()
                if len(nonterminals_left):
                    expand_location = nonterminals_left[-1]
                    next_rule_string = str(graph.node[expand_location])
                else:
                    next_rule_string = 'DONE'
            free_rules_left = self.MAX_LEN - self.t - 1 - \
                              self.grammar.terminal_distance(graph)

            this_mask = self.grammar.get_mask(next_rule_string, free_rules_left)
            masks.append(this_mask)

        self.t += 1
        return np.array(masks)

def hypergraphs_are_equivalent(graph1, graph2):
    def nodes_match(node1, node2):
        # parent nodes must be aligned
        if not graph1.is_parent_node(node1['node']) == graph2.is_parent_node(node2['node']):
            return False
        # and for all nodes the content must match
        return str(node1['node']) == str(node2['node'])
        # if node1['node'].is_terminal != node2['node'].is_terminal:
        #     return False
        # elif node1['node'].is_terminal:
        #     # for terminals (atoms), data must match
        #     return node1['node'].data == node2['node'].data
        # else:
        #
        #     if not graph1.is_parent_node(node1['node']) == graph2.is_parent_node(node2['node']):
        #         return False
        #     if not graph1.is_parent_node(node1['node']):
        #         return True # the isomorphy will take care of edge maching
        #     else: # parent nodes should match by ordered edge type, too!
        #         for edgeid1, edgeid2 in zip(node1['node'].edge_ids,node2['node'].edge_ids):
        #             if graph1.edges[edgeid1].type != graph2.edges[edgeid2].type:
        #                 return False
        #         return True

    def edges_match(edge1, edge2):
        return edge1['data'].type == edge2['data'].type

    from networkx.algorithms.isomorphism import GraphMatcher
    graph1_nx = graph1.to_nx()
    graph2_nx = graph2.to_nx()
    GM = GraphMatcher(graph1_nx, graph2_nx, edge_match=edges_match, node_match=nodes_match)

    if GM.is_isomorphic():
        # assert str(graph1) == str(graph2)
        return GM.mapping
        # do the edge id mapping
        # edge_mapping = {}
        # for edge1 in graph1_nx.edges:
        #     edge2 = graph2_nx.edges[(GM.mapping[edge1[0]],GM.mapping[edge1[1]])]
        #     edge_mapping[edge1['id']] = edge2['id']
        # return GM.mapping, edge_mapping
    else:
        return None




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
    def __init__(self, filename):
        self.grammar_filename = filename
        self.own_filename = 'init_'+ filename
        self.max_len = 0 # maximum observed number of rules so far
        self.last_processed = -1
        self.new_rules = []
        if os.path.isfile(filename):
            self.grammar = HypergraphGrammar.load(filename)
        else:
            self.grammar = HypergraphGrammar(cache_file=filename)

    def save(self):
        with open(self.own_filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(Class, filename):
        with open(filename, 'rb') as f:
            out = pickle.load(f)
        assert type(out) == Class
        return out

    def init_grammar(self, max_num_mols):
        L = []
        settings = get_data_location(molecules=True)
        # Read in the strings
        with open(settings['source_data'], 'r') as f:
            for line in f:
                line = line.strip()
                L.append(line)

        print('loaded data!')
        for ind, smiles in enumerate(L):
            if ind >= max_num_mols:
                break
            if ind > self.last_processed: # don't repeat
                try:
                    # this causes g to remember all the rules occurring in these molecules
                    these_actions = self.grammar.raw_strings_to_actions([smiles])
                    new_max_len = max([len(x) for x in these_actions])
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

        return self.max_len # maximum observed molecule length