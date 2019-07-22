from math import floor

import numpy as np

from generative_playground.codec.hypergraph import replace_nonterminal, HyperGraph, expand_index_to_id


class HypergraphMaskGenerator:
    def __init__(self, max_len, grammar, priors=True):
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


    def get_one_mask(self, graph, expand_id):
        steps_left = self.MAX_LEN - self.t
        grammar = self.grammar
        this_mask = get_one_mask_fun(grammar, steps_left, graph, expand_id)
        # if graph is None:
        #     next_rule_string = 'None'
        # else:
        #     if expand_id is not None:
        #         next_rule_string = str(graph.node[expand_id])
        #     else: # we're out of nonterminals, just use the padding rule
        #         next_rule_string = 'DONE'
        #
        # free_rules_left = steps_left - 1 - \
        #                   grammar.terminal_distance(graph)
        #
        # this_mask = grammar.get_mask(next_rule_string, free_rules_left)
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

        full_logit_priors, node_priors, node_mask = get_full_logit_priors(self.grammar, self.MAX_LEN - self.t, self.graphs)

        if self.priors == True:
            full_logit_priors += self.grammar.get_log_frequencies()[None, None, :]
        elif self.priors == 'conditional':
            full_logit_priors += self.get_log_conditional_frequencies()

        return self.graphs, node_priors, full_logit_priors  # already add in the node priors

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
                new_graph = apply_one_action(self.grammar, graph, last_act, exp_loc)
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
        return get_log_conditional_freqs(self.grammar, self.graphs)

    def action_prior_logits(self):
        masks = self.valid_action_mask()
        out = -1e4*(1-np.array(masks))
        if self.priors is True:
            out += self.grammar.get_log_frequencies()
        elif self.priors == 'conditional': # TODO this bit is broken, needs fixing
            all_cond_freqs = self.get_log_conditional_frequencies()
            if self.graphs[0] is None: # first step
                out += all_cond_freqs[:,0,:]
            else:
                cf = []
                for g, (graph, expand_loc) in enumerate(zip(self.graphs, self.next_expand_location)):
                    loc = graph.id_to_index(expand_loc)
                    cf.append(all_cond_freqs[g:g+1,loc,:])
                out += np.concatenate(cf, axis=0)
        return out

    def valid_node_mask(self):
        return valid_node_mask_fun(self.graphs)

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

def valid_node_mask_fun(graphs):
    max_nodes = max([1 if g is None else len(g) for g in graphs])
    out = np.zeros((len(graphs), max_nodes))
    for g, graph in enumerate(graphs):
        if graph is None:  # no graph, return value will be used as a mask but the result ignored
            out[g, :] = 1
        else:  # have a partially expanded graph, look for remaining nonterminals
            assert isinstance(graph, HyperGraph)
            child_ids = graph.child_ids()
            for n, node_id in enumerate(graph.node.keys()):
                if node_id in child_ids and n < max_nodes:
                    out[g, n] = 1
    return out


def apply_one_action(grammar, graph, last_action, expand_id):
    # TODO: allow None action
    last_rule = grammar.rules[last_action]
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

def get_log_conditional_freqs(grammar, graphs):
    freqs = -3*np.ones((len(graphs),
                       max([1 if g is None else len(g) for g in graphs]),
                       len(grammar)))
    for g, graph in enumerate(graphs):
        if graph is None: # first step, no graph yet
            query = (None, None)
            this_freqs = grammar.get_conditional_log_frequencies_single_query(query)
            # if we don't have a graph yet, put to every node the cond log frequencies of starting nodes
            for n in range(freqs.shape[1]):
                freqs[g, n, :] = this_freqs
        else:
            for n, node_id in enumerate(graph.node.keys()):
                if node_id in graph.child_ids():# conditional probabilities only matter for nonterminals
                    this_node = graph.node[node_id]
                    parent_rule = this_node.rule_id
                    nt_index = this_node.node_index # nonterminal index in the original rule
                    query = (parent_rule, nt_index)
                    this_freqs = grammar.get_conditional_log_frequencies_single_query(query)
                    freqs[g, n, :] = this_freqs

    return freqs

def mask_from_cond_tuple(grammar, cond_tuple):
    if cond_tuple[0] is None:
        rule_string = 'None'
    else:
        rule = grammar.rules[cond_tuple[0]]
        node_id = rule.index_to_id(cond_tuple[1])
        rule_string = get_rule_string(rule, node_id)
    this_mask = grammar.get_mask(rule_string, float('inf'))
    return this_mask

def get_rule_string(graph, expand_id):
    if graph is None:
        next_rule_string = 'None'
    else:
        if expand_id is not None:
            next_rule_string = str(graph.node[expand_id])
        else: # we're out of nonterminals, just use the padding rule
            next_rule_string = 'DONE'
    return next_rule_string

def get_one_mask_fun(grammar, steps_left, graph, expand_id):
    next_rule_string = get_rule_string(graph, expand_id)
    free_rules_left = steps_left - 1 - \
                      grammar.terminal_distance(graph)

    this_mask = grammar.get_mask(next_rule_string, free_rules_left)
    return this_mask

def action_to_condition(grammar, graph, action):
    """
    Converts an action on a graph to the conditioning tuple of the node implied by the action
    :param grammar:
    :param graph:
    :param action:
    :return:
    """
    pass

def get_full_logit_priors(grammar, steps_left, graphs):
    node_mask = valid_node_mask_fun(graphs)
    full_logit_priors = -1e5 * np.ones(list(node_mask.shape) + [len(grammar)])
    for g, graph in enumerate(graphs):
        if graph is None:  # first step, no graphs created yet
            full_logit_priors[g, 0, :] = -1e5 * (1 - np.array(get_one_mask_fun(grammar, steps_left, None, None)))
        elif len(graph.child_ids()) > 0:
            for l, loc in enumerate(graph.node.keys()):
                if loc in graph.child_ids():
                    full_logit_priors[g, l, :] = -1e5 * (1 - np.array(get_one_mask_fun(grammar, steps_left, graph, loc)))
        else:  # no nonterminals left, got to use the padding rule
            for l, loc in enumerate(graph.node.keys()):
                if loc != graph.parent_node_id:
                    full_logit_priors[g, l, :] = -1e5 * (1 - np.array(get_one_mask_fun(grammar, steps_left, graph, None)))
                    node_mask[g, l] = 1.0
                    break

    node_priors = -1e5 * (1 - np.array(node_mask))
    full_logit_priors += node_priors[:, :, None] # already add in the node priors so we know which nodes are impossible
    # TODO: instead, just return probabilities for nonterminal nodes?
    return full_logit_priors, node_priors, node_mask


def action_to_node_rule_indices(action_index, num_rules):
    node_index = floor(action_index / num_rules)
    rule_index = action_index % num_rules
    return node_index, rule_index