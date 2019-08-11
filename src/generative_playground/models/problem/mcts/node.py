import numpy as np

from generative_playground.codec.hypergraph import expand_index_to_id
from generative_playground.codec.hypergraph_mask_generator import get_log_conditional_freqs, get_full_logit_priors, \
    action_to_node_rule_indices, apply_one_action
from generative_playground.models.problem.mcts.result_repo import RuleChoiceRepository

class GlobalParameters:
    def __init__(self,
                 grammar,
                 max_depth,
                 exp_repo,
                 decay=0.99,
                 updates_to_refresh=100,
                 reward_fun=None):
        self.grammar = grammar
        self.max_depth = max_depth
        self.experience_repository = exp_repo
        self.decay = decay
        self.updates_to_refresh = updates_to_refresh
        self.reward_fun = reward_fun


class MCTSNode:
    def __init__(self,
                 global_params,
                 parent,
                 source_action,
                 depth,
                 reward_proc=None):
        """
        Creates a placeholder with just the value distribution guess, to be aggregated by the parent
        :param value_distr:
        """
        self.globals = global_params

        self.parent = parent
        self.source_action = source_action
        self.depth = depth
        self.reward_proc = reward_proc
        self.updates_since_refresh = 0

        self.value_distr_total = None
        self.child_run_count = 0

        if self.parent is not None:
            self.graph = apply_rule(parent.graph, source_action, global_params.grammar)
        else:
            self.graph = None

        if not graph_is_terminal(self.graph):
            self.children = [None for _ in range(len(global_params.grammar) *
                                                 (1 if self.graph is None else len(
                                                     self.graph)))]  # maps action index to child
            self.log_priors = None
            self.log_action_probs = None
            self.result_repo = None

            self.init_action_probabilities()

    def is_terminal(self):
        return graph_is_terminal(self.graph)

    def action_probabilities(self):
        if self.updates_since_refresh > self.globals.updates_to_refresh:
            self.refresh_probabilities()
        self.updates_since_refresh += 1
        return self.probs

    def apply_action(self, action):
        if self.children[action] is None:
            self.make_child(action)
        chosen_child = self.children[action]
        reward, info = get_reward(chosen_child.graph, self.globals.reward_fun)
        return chosen_child, reward, info

    def make_child(self, action):
        self.children[action] = MCTSNode(global_params=self.globals,
                                         parent=self,
                                         source_action=action,
                                         depth=self.depth + 1,
                                         reward_proc=self.reward_proc)

    def back_up(self, reward):
        # self.process_back_up(reward)
        if self.parent is not None:
            # update the local result cache
            self.parent.result_repo.update(self.source_action, reward)
            # update the global result cache
            self.globals.experience_repository.update(self.parent.graph, self.source_action, reward)
            self.parent.back_up(reward)

    def value_distr(self):
        return self.value_distr_total / self.child_run_count

    def generate_probs_from_log_action_probs(self):
        total_prob = self.log_action_probs + self.log_priors
        assert total_prob.max() > -1e4, "We need at least one allowed action!"
        tmp = np.exp(total_prob - total_prob.max())
        self.probs = tmp / tmp.sum()
        assert not np.isnan(self.probs.sum())

    # specific
    def refresh_probabilities(self):
        self.log_action_probs = self.result_repo.get_conditional_log_probs()
        self.generate_probs_from_log_action_probs()
        self.updates_since_refresh = 0

    def init_action_probabilities(self):
        self.log_action_probs = self.globals.experience_repository.get_log_probs_for_graph(self.graph).reshape((-1,))

        log_freqs = get_log_conditional_freqs(self.globals.grammar, [self.graph])

        # the full_logit_priors at this stage merely store the information about which actions are allowed
        full_logit_priors, _, node_mask = get_full_logit_priors(self.globals.grammar,
                                                                self.globals.max_depth - self.depth,
                                                                [self.graph])  # TODO: proper arguments
        priors = full_logit_priors + log_freqs
        self.log_priors = priors.reshape((-1,))
        self.generate_probs_from_log_action_probs()

        self.result_repo = RuleChoiceRepository(len(self.log_priors),
                                                reward_proc=self.reward_proc,
                                                mask=self.log_priors > -1e4,
                                                decay=self.globals.decay)


def get_reward(graph, reward_fun):
    if graph_is_terminal(graph):
        smiles = graph.to_smiles()
        return reward_fun([smiles])[0], {'smiles': smiles}
    else:
        return 0.0, None


def apply_rule(graph, action_index, grammar):
    '''
    Applies a rule to a graph, using the grammar to convert from index to actual rule
    :param graph: a hypergraph
    :param action_index:
    :return:
    '''
    node_index, rule_index = action_to_node_rule_indices(action_index, len(grammar))
    next_expand_id = expand_index_to_id(graph, node_index)
    new_graph = apply_one_action(grammar, graph, rule_index, next_expand_id)
    return new_graph


def graph_is_terminal(graph):
    # can't make this a method of HyperGraph because it must also work for None graphs
    # a cleaner way of doing that would be for HyperGraph to support empty graphs
    if graph is None:
        return False
    else:
        return len(graph.child_ids()) == 0