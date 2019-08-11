import math
from collections import OrderedDict

import numpy as np

from generative_playground.codec.hypergraph import expand_index_to_id, conditoning_tuple
from generative_playground.codec.hypergraph_grammar import HypergraphGrammar
from generative_playground.codec.hypergraph_mask_generator import action_to_node_rule_indices, get_one_mask_fun, \
    mask_from_cond_tuple


class ExperienceRepository:
    """
    This generates the starting probabilities for a newly created child in an MCTS graph
    The below implementation is meant for an explicit formula based on the experiences so far
    another version could instead train and sample a model deepQ-style
    """

    def __init__(self,
                 grammar: HypergraphGrammar,
                 reward_preprocessor=None,
                 rule_choice_repo_factory=None,
                 decay=0.99,
                 num_updates_to_refresh=1000,
                 conditional_keys=None):
        self.grammar = grammar
        self.reward_preprocessor = reward_preprocessor
        self.rule_choice_repo_factory = rule_choice_repo_factory
        self.conditional_keys = conditional_keys
        self.conditional_rewards = OrderedDict([(key, None) for key in conditional_keys])
        self.rewards_by_action = rule_choice_repo_factory(np.ones([len(grammar)]))#RuleChoiceRepository(self.reward_preprocessor,
                                                      # np.ones([len(grammar)]),
                                                      # decay=decay)
        # self.mask_by_cond_tuple = OrderedDict([(key, None) for key in grammar.conditional_frequencies.keys()])
        self.cond_tuple_to_index = {key: i for i, key in enumerate(conditional_keys)}
        self.decay = decay
        self.num_updates_to_refresh = num_updates_to_refresh

        self.refresh_conditional_log_probs()
        self.updates_since_last_refresh = 0

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

        self.conditional_store(cond_tuple).update(rule_ind, reward)
        self.updates_since_last_refresh += 1
        if self.updates_since_last_refresh > self.num_updates_to_refresh:
            self.refresh_conditional_log_probs()

    def conditional_store(self, cond_tuple):
        # if self.mask_by_cond_tuple[cond_tuple] is None:
        #     self.mask_by_cond_tuple[cond_tuple] = mask_from_cond_tuple(self.grammar, cond_tuple)
        if self.conditional_rewards[cond_tuple] is None:
            mask_by_cond_tuple = mask_from_cond_tuple(self.grammar, cond_tuple)
            self.conditional_rewards[cond_tuple] = self.rule_choice_repo_factory(mask_by_cond_tuple)
                # RuleChoiceRepository(self.reward_preprocessor,
                #                                                         mask_by_cond_tuple,
                #                                                         decay=self.decay)
        return self.conditional_rewards[cond_tuple]

    def refresh_conditional_log_probs(self):
        # collect masks, avg rewards and weights for every conditioning, actions for every cond_tuple
        masks = []
        reward_totals = []
        for cond_tuple in self.grammar.conditional_frequencies.keys():
            this_cache = self.conditional_store(cond_tuple)
            masks.append(~(this_cache.bool_mask))
            # enrich each conditional distribution with its rule index
            reward_totals += [(i,x) for i, x in enumerate(this_cache.reward_totals) if not this_cache.bool_mask[i]]
        # calculate regularized rewards
        # first pass calculates the global average
        wts, rewards = 0, 0.0
        for rule_ind, (wt, reward) in reward_totals:
            if reward is not None:
                wts += wt
                rewards += reward
        all_log_probs = np.zeros((len(self.conditional_keys) * len(self.grammar)))
        rewards_by_rule = self.rewards_by_action.reward_totals
        if wts > 0:
            avg_reward = rewards / wts
            reg_rewards = [regularize_reward(wt, reward, avg_reward, self.decay, rewards_by_rule[rule_ind])
                           for (rule_ind, (wt, reward)) in reward_totals]
            # call thompsom prob
            log_probs, probs = log_thompson_probabilities(np.array(reg_rewards))
            # cache the results per cond tuple
            all_masks = np.concatenate(masks)
            all_log_probs[all_masks] = log_probs
        elif wts < 0:
            raise ValueError("Wts should never be negative!")

        all_log_probs = all_log_probs.reshape((len(self.conditional_keys),
                                               len(self.grammar)))
        self.log_probs = all_log_probs
        self.updates_since_last_refresh = 0

    def get_log_probs_for_graph(self, graph):
        if graph is None:
            cond_tuple = (None, None)
            return self.log_probs[self.cond_tuple_to_index[cond_tuple], :]
        else:
            out = -1e5 * np.ones((len(graph), len(self.grammar)))
            child_ids = set(graph.child_ids())

            for n, node_id in enumerate(graph.node.keys()):
                if node_id in child_ids:
                    cond_tuple = conditoning_tuple(graph, node_id)
                    out[n] = self.log_probs[self.cond_tuple_to_index[cond_tuple], :]
            out.reshape((-1,))

            assert out.max() > -1e4, "At least some actions must be allowed"
            return out


class RuleChoiceRepository:
    """
    Stores the results of rule choices for a certain kind of starting node, generates sampling probabilities from these
    """

    def __init__(self, reward_proc, mask, decay=0.99):
        """
        :param reward_proc: converts incoming rewards into something we want to store, such as prob vectors
        :param mask: a vector of length num_rules, 1 for legal rules, 0 for illegal ones
        :param decay:
        """
        num_rules = len(mask)
        self.reward_preprocessor = reward_proc
        self.decay = decay
        self.bool_mask = np.array(mask, dtype=np.int32) == 0
        self.reward_totals = [[0, None] for _ in range(
            num_rules)]  # rewards can be stored as numbers or other objects supporting + and *, don't prejudge
        self.all_reward_totals = [0, None]

    def update(self, rule_ind, reward):
        self.reward_totals[rule_ind] = update_node(self.reward_totals[rule_ind], reward, self.decay,
                                                   self.reward_preprocessor)
        self.all_reward_totals = update_node(self.all_reward_totals, reward, self.decay,
                                             self.reward_preprocessor)

    def get_regularized_reward(self, rule_ind):
        # for actions that haven't been visited often, augment their reward with the average one for regularization
        max_wt = 1 / (1 - self.decay)
        assert self.all_reward_totals[
                   1] is not None, "This function should only be called after at least one experience!"
        avg_reward = self.avg_reward()
        assert avg_reward.sum() < float('inf')
        if self.reward_totals[rule_ind][1] is None:
            return avg_reward
        else:
            this_wt, this_reward = self.reward_totals[rule_ind]
            reg_wt = max(0, max_wt - this_wt)
            reg_reward = (this_reward + avg_reward * reg_wt) / (this_wt + reg_wt)
            assert reg_reward.sum() < float('inf')
            return reg_reward

    def avg_reward(self, rule_ind=None):
        if rule_ind is None or self.reward_totals[rule_ind][0] <= 0:
            return self.all_reward_totals[1] / self.all_reward_totals[0]
        else:
            return self.reward_totals[rule_ind][1]/self.reward_totals[rule_ind][0]


    def get_mask_and_rewards(self):
        return self.bool_mask, self.regularized_rewards()

    def get_conditional_log_probs(self):
        probs, avg_reward = self.get_conditional_log_probs_with_reward()
        return probs

    def regularized_rewards(self):
        reg_rewards = np.array([self.get_regularized_reward(rule_ind) for rule_ind in range(len(self.reward_totals))
                                if not self.bool_mask[rule_ind]])
        return reg_rewards

    def get_conditional_log_probs_with_reward(self):
        num_actions = len(self.reward_totals)
        out_log_probs = np.zeros(num_actions)
        out_log_probs[self.bool_mask] = -1e5  # kill the prohibited rules

        if self.all_reward_totals[1] is None:  # no experiences yet
            return out_log_probs, None
        else:
            all_rewards = self.regularized_rewards()
            log_probs, probs = log_thompson_probabilities(all_rewards)
            avg_reward = (all_rewards * probs.reshape((-1, 1))).sum(0)
            out_log_probs[~self.bool_mask] = log_probs
            return out_log_probs, avg_reward


def regularize_reward(this_wt, this_total_reward, avg_reward, decay, reward_by_rule=None):
    rule_wt, rule_total = reward_by_rule
    if rule_total is not None:
        rule_avg = rule_total/rule_wt
    else:
        rule_avg = 0

    max_wt = 1 / (1 - decay)
    assert avg_reward.sum() < float('inf')
    # handle the sparse cases
    if this_total_reward is None:
        if rule_total is None:
            return avg_reward
        else:
            return rule_avg
    else:
        # top up actuals first with observations by rule, then with global average
        combined_reg_wt = max(0, max_wt - this_wt)
        used_rule_wt = min(combined_reg_wt, rule_wt)
        used_avg_wt = max(0, combined_reg_wt-used_rule_wt)
        assert this_wt + used_rule_wt + used_avg_wt == max_wt
        reg_reward = (this_total_reward + rule_avg*used_rule_wt + avg_reward * used_avg_wt) / max_wt
        assert reg_reward.sum() < float('inf')
        return reg_reward


def to_bins(reward, num_bins):  # TODO: replace with a ProbabilityDistribution object
    reward = max(min(reward, 1.0), 0.0)
    out = np.zeros([num_bins])
    ind = math.floor(reward * num_bins * 0.9999)
    out[ind] = 1
    return out


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


def log_thompson_probabilities(ps):
    """
    Calculate thompson probabilities for one slice, that's already been masked
    :param ps: actions x bins floats reward probability distribution per action
    :return: actions floats
    """
    ps += 1e-5  # to guarantee positivity
    ps = ps / ps.sum(axis=1, keepdims=True)
    log_ps = np.log(ps)
    log_cdfs = np.log(ps.cumsum(axis=1))
    log_cdf_prod = np.sum(log_cdfs, axis=0, keepdims=True)
    pre_thompson = (log_ps[:, 1:] + log_cdf_prod[:, :-1] - log_cdfs[:, :-1])
    thompson = (np.exp(pre_thompson - pre_thompson.max())).sum(1)
    log_thompson = np.log(thompson + 1e-8)
    # thompson2 = (ps*cdf_prod / cdfs).sum(1) # this should always be >1
    total = thompson.sum()
    assert total > 0, "total probs must be positive!"
    out = thompson / total
    assert not np.isnan(log_thompson.sum()), "Something went wrong in thompson probs, got a nan"
    return log_thompson, out


def softmax_probabilities(log_ps):
    '''
    Calculate thompson probabilities for one slice, that's already been masked
    :param ps: actions x bins floats
    :return: actions floats
    '''
    ps = np.exp(log_ps - log_ps.max())
    ps = ps / ps.sum(axis=1, keepdims=True)
    evs = ps.cumsum(axis=1).sum(axis=1)  # a very crude way to do expected value :)
    evs = (evs - evs.min() + 1e-5) / (evs.max() - evs.min() + 1e-5)
    evs = evs / evs.sum()
    return np.log(evs)
