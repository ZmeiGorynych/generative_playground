import numpy as np
import uuid
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
                 reward_fun=None,
                 reward_proc=None,
                 rule_choice_repo_factory=None,
                 state_store={}):
        self.grammar = grammar
        self.max_depth = max_depth
        self.experience_repository = exp_repo
        self.decay = decay
        self.updates_to_refresh = updates_to_refresh
        self.reward_fun = reward_fun
        self.reward_proc = reward_proc
        self.rule_choice_repo_factory = rule_choice_repo_factory
        self.state_store = state_store

class LocalState:
    def __init__(self):
        pass


class MCTSNodeParent:
    def __init__(self,
                 global_params,
                 parent,
                 source_action,
                 depth):
        """
        Creates a placeholder with just the value distribution guess, to be aggregated by the parent
        :param value_distr:
        """
        self.id = uuid.uuid4()
        global_params.state_store[self.id] = LocalState()

        self.globals = global_params

        self.parent = parent
        self.source_action = source_action
        self.depth = depth
        self.updates_since_refresh = 0

        if self.parent is not None:
            self.locals().graph = apply_rule(self.parent.locals().graph, source_action, global_params.grammar)
        else:
            self.locals().graph = None

        if not graph_is_terminal(self.locals().graph):
            self.children = [None for _ in range(len(global_params.grammar) *
                                                 (1 if self.locals().graph is None else len(
                                                     self.locals().graph)))]  # maps action index to child
            # the full_logit_priors at this stage merely store the information about which actions are allowed
            full_logit_priors, _, node_mask = get_full_logit_priors(self.globals.grammar,
                                                                    self.globals.max_depth - self.depth,
                                                                    [self.locals().graph])
            log_freqs = get_log_conditional_freqs(self.globals.grammar, [self.locals().graph])
            # these are the emprirical priors from the original dataset
            self.locals().log_priors = (full_logit_priors + log_freqs).reshape((-1,))

            mask = full_logit_priors.reshape([-1]) > -1e4
            self.locals().result_repo = self.globals.rule_choice_repo_factory(mask)

            self.locals().log_action_probs = None
            self.locals().probs = None

            self.init_action_probabilities()
        assert hasattr(self.locals(), 'graph')

    def locals(self):
        return self.globals.state_store[self.id]

    def is_terminal(self):
        return graph_is_terminal(self.locals().graph)

    def action_probabilities(self):
        if self.updates_since_refresh > self.globals.updates_to_refresh:
            self.refresh_probabilities()
        self.updates_since_refresh += 1
        return self.locals().probs

    def apply_action(self, action):
        if self.children[action] is None:
            self.children[action] = self.make_child(action)
        chosen_child = self.children[action]
        reward, info = get_reward(chosen_child.locals().graph, self.globals.reward_fun)
        return chosen_child, reward, info

    def make_child(self, action):
        return type(self)(global_params=self.globals,
                                           parent=self,
                                           source_action=action,
                                           depth=self.depth + 1)

    def back_up(self, reward):
        # self.process_back_up(reward)
        if self.parent is not None:
            # update the local result cache
            self.parent.locals().result_repo.update(self.source_action, reward)
            # update the global result cache
            self.globals.experience_repository.update(self.parent.locals().graph, self.source_action, reward)
            self.parent.back_up(reward)

    def generate_probs_from_log_action_probs(self):
        """
        Helper function - combines the empirical priors with model-generated action probabilities
        :return:
        """
        total_prob = self.locals().log_action_probs + self.locals().log_priors
        assert total_prob.max() > -1e4, "We need at least one allowed action!"
        tmp = np.exp(total_prob - total_prob.max())
        self.locals().probs = tmp / tmp.sum()
        assert not np.isnan(self.locals().probs.sum())

    def refresh_probabilities(self):
        raise NotImplementedError

    def init_action_probabilities(self):
        raise NotImplementedError


class MCTSNodeLocalThompson(MCTSNodeParent):
    def init_action_probabilities(self):
        # gets called on node creation
        self.locals().log_action_probs = self.globals.experience_repository.get_log_probs_for_graph(self.locals().graph).reshape((-1,))
        self.generate_probs_from_log_action_probs()

    def refresh_probabilities(self):
        # gets called each time a node decides to refresh its probability vector
        self.locals().log_action_probs = self.locals().result_repo.get_conditional_log_probs()
        self.generate_probs_from_log_action_probs()
        self.updates_since_refresh = 0


class MCTSNodeGlobalThompson(MCTSNodeParent):
    """
    This version just does Thompson sampling by global conditioning, with no feedback from local experience
    """
    def init_action_probabilities(self):
        # gets called on node creation
        self.locals().log_action_probs = self.globals.experience_repository.get_log_probs_for_graph(self.locals().graph).reshape((-1,))
        self.generate_probs_from_log_action_probs()
        self.updates_since_refresh = 0

    def refresh_probabilities(self):
        # gets called each time a node decides to refresh its probability vector
        self.init_action_probabilities()

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
