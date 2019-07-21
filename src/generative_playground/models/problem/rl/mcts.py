import numpy as np
import math
from collections import OrderedDict
from generative_playground.codec.hypergraph_grammar import HypergraphGrammar
from generative_playground.codec.hypergraph_mask_generator import *
from generative_playground.codec.hypergraph import expand_index_to_id, conditoning_tuple
from generative_playground.codec.codec import get_codec
from generative_playground.molecules.guacamol_utils import guacamol_goal_scoring_functions


def explore(root_node, num_sims):
    for _ in range(num_sims):
        next_node = root_node
        reward = 0
        while True:
            is_terminal = next_node.is_terminal()
            if is_terminal:
                next_node.back_up(reward)
                break
            else:
                probs = next_node.action_probabilities()
                action = np.random.multinomial(1,probs).argmax()
                next_node, reward = next_node.apply_action(action)

class ExperienceRepository:
    """
    This generates the starting probabilities for a newly created child in an MCTS graph
    The below implementation is meant for an explicit formula based on the experiences so far
    another version could instead train and sample a model deepQ-style
    """
    def __init__(self,
                 grammar: HypergraphGrammar,
                 reward_preprocessor=lambda x: to_bins(x, num_bins),
                 decay=0.99):
        self.grammar = grammar
        self.reward_preprocessor = reward_preprocessor
        self.conditional_rewards = OrderedDict([(key, None) for key in grammar.conditional_frequencies.keys()])
        self.mask_by_cond_tuple = OrderedDict([(key, None) for key in grammar.conditional_frequencies.keys()])
        self.decay = decay

    def update(self, graph, action, reward):
        """
        Stores an experience tuple
        :param graph: The __starting__ graph to which an action was applied
        :param action: the full action int, combining node choice and rule choice
        :param reward: The actual reward, to be binned etc as necessary
        :return: nothing
        """
        node_ind, rule_ind = action_to_node_rule_indices(action, len(self.grammar))
        node_id = expand_index_to_id(graph, node_ind)
        cond_tuple = conditoning_tuple(graph, node_id)
        if self.mask_by_cond_tuple[cond_tuple] is None:
            self.mask_by_cond_tuple[cond_tuple] = get_one_mask_fun(self.grammar,
                                                                   1e6,
                                                                   graph,
                                                                   node_id)

        # for each conditioning tuple we must choose between rules, so cond_tuple and rule_ind are the two bits we care about
        if self.conditional_rewards[cond_tuple] is None:
            self.conditional_rewards[cond_tuple] = RuleChoiceRepository(len(self.grammar),
                                                                        self.reward_preprocessor,
                                                                        self.mask_by_cond_tuple[cond_tuple],
                                                                        decay=self.decay)
        self.conditional_rewards[cond_tuple].update(rule_ind, reward)

    def get_conditional_log_probs(self, cond_tuple):
        if self.conditional_rewards[cond_tuple] is not None:
            return self.conditional_rewards[cond_tuple].get_conditional_log_probs()
        else:
            return np.zeros(len(self.grammar))

    def get_log_probs_for_graph(self, graph):
        if graph is None:
            return self.get_conditional_log_probs((None, None))
        else:
            out = -1e5*np.ones((len(graph), len(self.grammar)))
            child_ids = set(graph.child_ids())
            for n, node_id in enumerate(graph.node.keys()):
                if node_id in child_ids:
                    cond_tuple = conditoning_tuple(graph, node_id)
                    out[n,:] = self.get_conditional_log_probs(cond_tuple)

            out = out.reshape((-1))
            assert out.max() > -1e4, "At least some actions must be allowed"
            return out


class RuleChoiceRepository:
    """
    Stores the results of rule choices for a certain kind of starting node, generates sampling probabilities from these
    """
    def __init__(self, num_rules, reward_proc, mask, decay=0.99):
        """

        :param num_rules: number of rules
        :param reward_proc: converts incoming rewards into something we want to store, such as prob vectors
        :param mask: a vector of length num_rules, 1 for legal rules, 0 for illegal ones
        :param decay:
        """
        self.grammar = grammar_name
        self.reward_preprocessor = reward_proc
        self.decay = decay
        self.bool_mask = np.array(mask, dtype=np.int32) == 0
        self.reward_totals = [[0, None] for _ in range(num_rules)] # rewards can be stored as numbers or other objects supporting + and *, don't prejudge
        self.all_reward_totals = [0, None]

    def update(self, rule_ind, reward):
        self.reward_totals[rule_ind] = update_node(self.reward_totals[rule_ind], reward, self.decay, self.reward_preprocessor)
        self.all_reward_totals = update_node(self.all_reward_totals, reward, self.decay, self.reward_preprocessor)

    def get_regularized_reward(self, rule_ind):
        # for actions that haven't been visited often, augment their reward with the average one for regularization
        max_wt = 1/(1-self.decay)
        assert self.all_reward_totals[1] is not None, "This function should only be called after at least one experience!"
        avg_reward = self.all_reward_totals[1]/self.all_reward_totals[0]
        assert avg_reward.sum() < float('inf')
        if self.reward_totals[rule_ind][1] is None:
            return avg_reward
        else:
            this_wt, this_reward = self.reward_totals[rule_ind]
            reg_wt = max(0, max_wt - this_wt)
            reg_reward = (this_reward + avg_reward*reg_wt)/(this_wt + reg_wt)
            assert reg_reward.sum() < float('inf')
            return reg_reward

    def get_conditional_log_probs(self):
        probs = np.zeros(len(self.reward_totals))
        probs[self.bool_mask] = -1e5 # kill the prohibited rules

        if self.all_reward_totals[1] is None: # no experiences yet
            return probs
        else:
            all_rewards = np.array([self.get_regularized_reward(rule_ind) for rule_ind in range(len(self.reward_totals))
                                    if not self.bool_mask[rule_ind]])
            th_probs = thompson_probabilities(all_rewards) # TODO: fix and insert the Thompson ones!
            probs[~self.bool_mask] = th_probs
            return probs

def update_node(node, reward, decay, reward_preprocessor):
    weight, proc_reward = reward_preprocessor(reward)
    if node[1] is None:
        node = [weight, proc_reward]
    else:
        node[0] *= (decay ** weight)
        node[0] += weight

        node[1] *= (decay ** weight)
        node[1] += proc_reward
    return node

# Tree:
# linked tree with nodes
class MCTSNode:
    def __init__(self, grammar, parent, source_action, max_depth, depth, exp_repo, decay=0.99):
        """
        Creates a placeholder with just the value distribution guess, to be aggregated by the parent
        :param value_distr:
        """
        self.grammar = grammar
        self.parent = parent
        self.source_action = source_action
        self.max_depth = max_depth
        self.depth = depth
        self.value_distr_total = None
        self.child_run_count = 0
        self.refresh_prob_thresh = 0.1
        self.experience_repository = exp_repo
        self.decay = decay

        if self.parent is not None:
            self.graph = apply_rule(parent.graph, source_action, grammar)
        else:
            self.graph = None

        if not graph_is_terminal(self.graph):
            self.log_action_probs = exp_repo.get_log_probs_for_graph(self.graph)

            self.children = [None for _ in range(len(grammar)*
                                                 (1 if self.graph is None else len(self.graph)))] # maps action index to child


            log_freqs = get_log_conditional_freqs(grammar, [self.graph])
            # the full_logit_priors at this stage merely store the information about which actions are allowed
            full_logit_priors, _, node_mask = get_full_logit_priors(self.grammar, self.max_depth - self.depth,
                                                                    [self.graph])  # TODO: proper arguments
            priors = full_logit_priors + log_freqs

            self.log_priors = priors.reshape((-1,))
            # self.priors = np.exp(self.log_priors - self.log_priors.max())
            self.refresh_probabilities()

    def is_terminal(self):
        return graph_is_terminal(self.graph)

    def value_distr(self):
        return self.value_distr_total/self.child_run_count

    def action_probabilities(self):
        if np.random.random([1][0]) < self.refresh_prob_thresh:
            self.refresh_probabilities()
        return self.probs

    # specific
    def refresh_probabilities(self): # TODO: replace the stub with  a real implementation
        total_prob = self.log_action_probs + self.log_priors
        assert total_prob.max() > -1e4, "We need at least one allowed action!"
        tmp = np.exp(total_prob - total_prob.max())
        self.probs = tmp/tmp.sum()
        assert not np.isnan(self.probs.sum())
        # # iterate over all children, get their value distrs, calculate Thompson probs
        # child_distributions = np.array([child.value_distr() for child in self.children])
        # self.action_probs = thompson_probabilities(child_distributions)

    def apply_action(self, action):
        if self.children[action] is None:
            self.make_child(action)
        chosen_child = self.children[action]
        reward = get_reward(chosen_child.graph)
        return chosen_child, reward

    def make_child(self, action):
        self.children[action] = MCTSNode(grammar=self.grammar,
                                         parent=self,
                                         source_action=action,
                                         max_depth=self.max_depth,
                                         depth=self.depth+1,
                                         exp_repo=self.experience_repository,
                                         decay=self.decay)

    def back_up(self, reward):
        self.process_back_up(reward)
        if self.parent is not None:
            self.experience_repository.update(self.parent.graph, self.source_action, reward)
            self.parent.back_up(reward)

    # this one is specific
    def process_back_up(self, reward):
        binned_reward = to_bins(reward, num_bins)
        if self.value_distr_total is None:
            self.value_distr_total = binned_reward
            self.child_run_count = 1
        else:
            self.value_distr_total *= self.decay
            self.value_distr_total += binned_reward

            self.child_run_count *= self.decay
            self.child_run_count += 1


def to_bins(reward, num_bins): # TODO: replace with a ProbabilityDistribution object
    reward = max(min(reward,1.0), 0.0)
    out = np.zeros([num_bins])
    ind = math.floor(reward*num_bins*0.9999)
    out[ind] = 1
    return out


def thompson_probabilities(log_ps):
    '''
    Calculate thompson probabilities for one slice, that's already been masked
    :param ps: actions x bins floats
    :return: actions floats
    '''
    ps = np.exp(log_ps - log_ps.max())
    ps = ps/ps.sum(axis=1, keepdims=True)
    cdfs = ps.cumsum(axis=1) + 1e-5
    cdf_prod = np.prod(cdfs, axis=0, keepdims=True)
    thompson = (ps[:, 1:] * cdf_prod[:, :-1] / cdfs[:, :-1]).sum(1)
    # thompson2 = (ps*cdf_prod / cdfs).sum(1) # this should always be >1
    total = thompson.sum()
    assert 0 < total <= 1
    return np.log(thompson/total)

def softmax_probabilities(log_ps):
    '''
    Calculate thompson probabilities for one slice, that's already been masked
    :param ps: actions x bins floats
    :return: actions floats
    '''
    ps = np.exp(log_ps - log_ps.max())
    ps = ps/ps.sum(axis=1, keepdims=True)
    evs = ps.cumsum(axis=1).sum(axis=1) # a very crude way to do expected value :)
    evs = (evs - evs.min()+1e-5)/(evs.max()-evs.min()+1e-5)
    evs = evs/evs.sum()
    return np.log(evs)

def get_reward(graph):
    if graph_is_terminal(graph):
        return reward_fun([graph.to_smiles()])[0]
    else:
        return 0.0


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

if __name__=='__main__':
    num_bins = 20  # TODO: replace with a Value Distribution object
    ver = 'trivial'
    obj_num = 0
    reward_fun = guacamol_goal_scoring_functions(ver)[obj_num]
    grammar_cache = 'hyper_grammar_guac_10k_with_clique_collapse.pickle'  # 'hyper_grammar.pickle'
    grammar_name = 'hypergraph:' + grammar_cache
    max_seq_length = 30
    codec = get_codec(True, grammar_name, max_seq_length)
    exp_repo = ExperienceRepository(grammar=codec.grammar,
                                    reward_preprocessor=lambda x: (1,to_bins(x, num_bins)),
                                    decay=0.99)
    root_node = MCTSNode(grammar=codec.grammar,
                         parent=None,
                         source_action=None,
                         max_depth=max_seq_length,
                         depth=1,
                         exp_repo=exp_repo,
                         decay=0.99)
    explore(root_node, 100)
    print(root_node.value_distr())
    print("done!")