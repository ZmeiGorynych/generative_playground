from collections import OrderedDict, defaultdict
from generative_playground.molecules.lean_settings import get_data_location, molecules_root_location
import frozendict
from generative_playground.codec.parent_codec import GenericCodec
from generative_playground.codec.hypergraph import (
    HyperGraph, HypergraphTree, replace_nonterminal, to_mol, MolToSmiles, MolFromSmiles, hypergraphs_are_equivalent
)
from generative_playground.codec.hypergraph_parser import hypergraph_parser, tree_with_rule_inds_to_list_of_tuples
from generative_playground.molecules.data_utils.zinc_utils import get_zinc_smiles
import networkx as nx
import pickle
import zipfile
import os, copy
import numpy as np
from pathlib import Path
import math

grammar_data_location = molecules_root_location + 'data/grammar/'


def full_location(filename):
    return os.path.realpath(grammar_data_location + filename)


class HypergraphGrammar(GenericCodec):
    def __init__(self, cache_file='tmp.pickle', max_len=None):
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

    def get_conditional_log_frequencies(self, x):
        out = np.zeros(len(self.rules))
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

        self._rule_term_dist_deltas = tuple(self._rule_term_dist_deltas)  # make it immutable

        assert min(self._rule_term_dist_deltas[1:]) >= 0
        assert len(self._rule_term_dist_deltas) == len(self.rules)

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


class HypergraphMaskGenerator:
    def __init__(self, max_len, grammar, priors=False):
        self.grammar = grammar
        self.MAX_LEN = max_len
        self.priors = priors
        self.graphs = None
        self.t = 0
        self.last_action = None
        self.next_expand_location = None

    def reset(self):
        self.graphs = None
        self.t = 0
        self.last_action = None
        self.next_expand_location = None

    def apply_one_action(self, graph, last_action, expand_id):
        # TODO: allow None action
        last_rule = self.grammar.rules[last_action]
        if graph is None:
            # first call, graph is being created
            assert last_rule.parent_node() is None, "The first rule in the sequence must have no parent node"
            graph = last_rule.clone()
        else:
            if expand_id is not None:
                graph = replace_nonterminal(graph,
                                            expand_id,
                                            last_rule)
            else:
                # no more nonterminals, nothing for us to do but assert rule validity
                assert last_rule is None, "Trying to expand a graph with no nonterminals" # the padding rule
        # validity check
        assert graph.parent_node_id is None
        return graph

    def get_one_mask(self, graph, expand_id):
        if graph is None:
            next_rule_string = 'None'
        else:
            if expand_id is not None:
                next_rule_string = str(graph.node[expand_id])
            else: # we're out of nonterminals, just use the padding rule
                next_rule_string = 'DONE'

        free_rules_left = self.MAX_LEN - self.t - 1 - \
                          self.grammar.terminal_distance(graph)

        this_mask = self.grammar.get_mask(next_rule_string, free_rules_left)
        return this_mask

    def __call__(self, last_action):
        self.apply_action(last_action)
        return self.valid_action_mask()

    def step(self, last_action):
        """
        A version for OpenAI gym-style interface, encapsulating node choice and action choice
        :param last_action:
        :return:
        """
        last_node, last_rule = last_action
        if last_node is not None:
            self.pick_next_node_to_expand(last_node) # converts index to node id, validating it in the process
        self.apply_action(last_rule)
        next_node = self.valid_node_mask()
        full_logit_priors = -1e5*np.ones(list(next_node.shape) + [len(self.grammar)])
        for g, graph in enumerate(self.graphs):
            if graph is None: # first step, no graphs created yet
                full_logit_priors[g, 0, :] = -1e5*(1-np.array(self.get_one_mask(None, None)))
            elif len(graph.child_ids()) > 0:
                for l, loc in enumerate(graph.node.keys()):
                    if loc in graph.child_ids():
                        full_logit_priors[g,l,:] = -1e5*(1-np.array(self.get_one_mask(graph, loc)))
            else: # no nonterminals left, got to use the padding rule
                for l, loc in enumerate(graph.node.keys()):
                    if loc != graph.parent_node_id:
                        full_logit_priors[g,l,:] = -1e5*(1-np.array(self.get_one_mask(graph, None)))
                        next_node[g, l] = 1.0
                        break

        log_freqs = self.grammar.get_log_frequencies()[None, None, :]
        full_logit_priors += log_freqs
        return self.graphs, -1e5*(1-np.array(next_node)), full_logit_priors



    def apply_action(self, last_action):
        self.last_action = last_action # to be used for next step's conditional frequencies
        if self.t >= self.MAX_LEN:
            raise StopIteration("maximum sequence length exceeded for decoder")

            # apply the last action
        if last_action[0] is None:
            assert self.t == 0 and self.graphs is None, "Trying to apply a None action to initialized graphs"
            # first call
            self.graphs = [None for _ in range(len(last_action))]

        else:
            # evaluate the rule; assume the rule is valid
            for ind, graph, last_act, exp_loc in \
                    zip(range(len(self.graphs)), self.graphs, last_action, self.next_expand_location):
                # exp_loc_id = expand_index_to_id(graph, exp_loc)
                new_graph = self.apply_one_action(graph, last_act, exp_loc)
                self.graphs[ind] = new_graph
        # reset the expand locations
        self.next_expand_location = [expand_index_to_id(graph, None) for graph in self.graphs]
        self.t += 1

    def valid_action_mask(self):
        masks = []
        for graph, expand_loc in zip(self.graphs, self.next_expand_location):
            this_mask = self.get_one_mask(graph, expand_loc)
            masks.append(this_mask)
        out = np.array(masks)#
        return out

    def get_log_conditional_frequencies(self):
        freqs = []
        for graph, expand_loc in zip(self.graphs, self.next_expand_location):
            if graph is None: # first step
                query = (None, None)
            elif expand_loc is None: # graph is finished, we're just doing padding
                assert len(graph.child_ids()) == 0
                return np.zeros((len(self.graphs), len(self.grammar.rules)))
            else:
                this_node = graph.node[expand_loc]
                parent_rule = this_node.rule_id
                nt_index = this_node.node_index # nonterminal index in teh original rule
                query = (parent_rule, nt_index)
            this_freqs = self.grammar.get_conditional_log_frequencies(query)
            freqs.append(this_freqs)
        out = np.array(freqs)
        return out

    def action_prior_logits(self):
        masks = self.valid_action_mask()
        out = -1e4*(1-np.array(masks))
        if self.priors is True:
            out += self.grammar.get_log_frequencies()
        elif self.priors == 'conditional':
            out += self.get_log_conditional_frequencies()
        return out

    def valid_node_mask(self):
        max_nodes = max([1 if g is None else len(g) for g in self.graphs])
        out = np.zeros((len(self.graphs), max_nodes))
        for g, graph in enumerate(self.graphs):
            if graph is None: # no graph, return value will be used as a mask but the result ignored
                out[g,:] = 1
            else: # have a partially expanded graph, look for remaining nonterminals
                assert isinstance(graph, HyperGraph)
                child_ids = graph.child_ids()
                for n, node_id in enumerate(graph.node.keys()):
                    if node_id in child_ids and n < max_nodes:
                        out[g, n] = 1
        return out


    def pick_next_node_to_expand(self, node_idx):
        '''
        Sets the specific nonterminals to expand in the next step
        :param node_idx:
        :return: None
        '''
        if self.next_expand_location is None:
            self.next_expand_location = [None for _ in range(len(node_idx))]
        assert len(node_idx) == len(self.graphs), "Wrong number of node locations"
        for i, graph, node_index in zip(range(len(self.graphs)), self.graphs, node_idx):
            if graph is not None: # during the very first step graph is None, before first rule was picked
                self.next_expand_location[i] = expand_index_to_id(graph, node_index)

def expand_index_to_id(graph, expand_index=None):
    '''

    :param graph: HyperGraph or None for the first step
    :param expand_index: None or an index of the node in graph.node that we want to expand next
    :return: id of the next node to expand, or None if we have nothing left to expand
    '''
    if graph is None:
        return None # expand_id will be ignored as we'll be starting a new graph

    nonterminals_left = graph.nonterminal_ids()
    if len(nonterminals_left):
        if expand_index is None:
            expand_id = nonterminals_left[-1]
        else:
            expand_id = list(graph.node.keys())[expand_index]
            assert expand_id in nonterminals_left, \
                "The proposed expand location is not valid"
    else:
        expand_id = None
    return expand_id


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
        L = get_zinc_smiles(max_num_mols)
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
        self.grammar.calc_terminal_distance()
        return self.max_len # maximum observed molecule length
