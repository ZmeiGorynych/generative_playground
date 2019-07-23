import numpy as np
import math
from collections import OrderedDict
from generative_playground.codec.hypergraph_grammar import HypergraphGrammar
from generative_playground.codec.hypergraph_mask_generator import *
from generative_playground.codec.hypergraph import expand_index_to_id, conditoning_tuple
from generative_playground.codec.codec import get_codec
from generative_playground.molecules.guacamol_utils import guacamol_goal_scoring_functions


def explore(root_node, num_sims):
    rewards = []
    infos = []
    for _ in range(num_sims):
        next_node = root_node
        while True:
            probs = next_node.action_probabilities()
            action = np.random.multinomial(1,probs).argmax()
            next_node, reward, info = next_node.apply_action(action)
            is_terminal = next_node.is_terminal()
            if is_terminal:
                next_node.back_up(reward)
                rewards.append(reward)
                infos.append(info)
                break
    return rewards, infos

class ExperienceRepository:
    """
    This generates the starting probabilities for a newly created child in an MCTS graph
    The below implementation is meant for an explicit formula based on the experiences so far
    another version could instead train and sample a model deepQ-style
    """
    def __init__(self,
                 grammar: HypergraphGrammar,
                 reward_preprocessor=lambda x: to_bins(x, num_bins),
                 decay=0.99,
                 num_updates_to_refresh=1000):
        self.grammar = grammar
        self.reward_preprocessor = reward_preprocessor
        self.conditional_rewards = OrderedDict([(key, None) for key in grammar.conditional_frequencies.keys()])
        self.mask_by_cond_tuple = OrderedDict([(key, None) for key in grammar.conditional_frequencies.keys()])
        self.cond_tuple_to_index = {key: i for i, key in enumerate(grammar.conditional_frequencies.keys())}
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
        self.conditional_store(cond_tuple).update(rule_ind, reward)
        self.updates_since_last_refresh += 1
        if self.updates_since_last_refresh > self.num_updates_to_refresh:
            self.refresh_conditional_log_probs()

    def conditional_store(self, cond_tuple):
        if self.mask_by_cond_tuple[cond_tuple] is None:
            self.mask_by_cond_tuple[cond_tuple] = mask_from_cond_tuple(self.grammar, cond_tuple)
        if self.conditional_rewards[cond_tuple] is None:
            self.conditional_rewards[cond_tuple] = RuleChoiceRepository(len(self.grammar),
                                                                        self.reward_preprocessor,
                                                                        self.mask_by_cond_tuple[cond_tuple],
                                                                        decay=self.decay)
        return self.conditional_rewards[cond_tuple]


    def refresh_conditional_log_probs(self):
        # collect masks, avg rewards and weights for every conditioning, actions for every cond_tuple
        masks = []
        reward_totals = []
        for cond_tuple in self.grammar.conditional_frequencies.keys():
            this_cache = self.conditional_store(cond_tuple)
            masks.append(~(this_cache.bool_mask))
            reward_totals +=[x for i,x in enumerate(this_cache.reward_totals) if not this_cache.bool_mask[i]]
        # calculate regularized rewards
        # first pass calculates the global average
        wts, rewards = 0, 0.0
        for wt, reward in reward_totals:
            if reward is not None:
                wts += wt
                rewards += reward
        all_log_probs = np.zeros((len(self.grammar.conditional_frequencies)*len(self.grammar)))

        if wts > 0:
            avg_reward = rewards/wts
            reg_rewards = [regularize_reward(wt, reward, avg_reward, self.decay) for (wt, reward) in reward_totals]
            # call thompsom prob
            log_probs, probs = log_thompson_probabilities(np.array(reg_rewards))
            # cache the results per cond tuple
            all_masks = np.concatenate(masks)
            all_log_probs[all_masks] = log_probs
        elif wts < 0:
            raise ValueError("Wts should never be negative!")

        all_log_probs = all_log_probs.reshape((len(self.grammar.conditional_frequencies),
                                               len(self.grammar)))
        self.log_probs = all_log_probs
        self.updates_since_last_refresh = 0


    def get_log_probs_for_graph(self, graph):
        if graph is None:
            cond_tuple = (None,  None)
            return self.log_probs[self.cond_tuple_to_index[cond_tuple],:]
        else:
            out = -1e5*np.ones((len(graph),len(self.grammar)))
            child_ids = set(graph.child_ids())

            for n, node_id in enumerate(graph.node.keys()):
                if node_id in child_ids:
                    cond_tuple = conditoning_tuple(graph, node_id)
                    out[n] = self.log_probs[self.cond_tuple_to_index[cond_tuple],:]
            out.reshape((-1,))

            assert out.max() > -1e4, "At least some actions must be allowed"
            return out

def child_rewards_to_log_probs(child_rewards):
    total = 0
    counter = 0
    for c in child_rewards:
        if c is not None:
            total+=c
            counter += 1
    if counter == 0:
        return np.zeros((len(child_rewards)))
    avg_reward = total/counter
    for c in range(len(child_rewards)):
        if child_rewards[c] is None:
            child_rewards[c] = avg_reward

    log_probs, probs = log_thompson_probabilities(np.array(child_rewards))
    return log_probs


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
        avg_reward = self.avg_reward()
        assert avg_reward.sum() < float('inf')
        if self.reward_totals[rule_ind][1] is None:
            return avg_reward
        else:
            this_wt, this_reward = self.reward_totals[rule_ind]
            reg_wt = max(0, max_wt - this_wt)
            reg_reward = (this_reward + avg_reward*reg_wt)/(this_wt + reg_wt)
            assert reg_reward.sum() < float('inf')
            return reg_reward

    def avg_reward(self):
        return self.all_reward_totals[1]/self.all_reward_totals[0]

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
        out_log_probs[self.bool_mask] = -1e5 # kill the prohibited rules

        if self.all_reward_totals[1] is None: # no experiences yet
            return out_log_probs, None
        else:
            all_rewards = self.regularized_rewards()
            log_probs, probs = log_thompson_probabilities(all_rewards)
            avg_reward = (all_rewards*probs.reshape((-1,1))).sum(0)
            out_log_probs[~self.bool_mask] = log_probs
            return out_log_probs, avg_reward

def regularize_reward(this_wt, this_total_reward, avg_reward, decay):
    max_wt = 1 / (1 - decay)
    assert avg_reward.sum() < float('inf')
    if this_total_reward is None:
        return avg_reward
    else:
        reg_wt = max(0, max_wt - this_wt)
        reg_reward = (this_total_reward + avg_reward * reg_wt) / (this_wt + reg_wt)
        assert reg_reward.sum() < float('inf')
        return reg_reward

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
    def __init__(self, grammar, parent, source_action,
                 max_depth, depth, exp_repo,
                 decay=0.99,
                 reward_proc=None,
                 refresh_prob_thresh=0.01):
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
        self.refresh_prob_thresh = refresh_prob_thresh
        self.experience_repository = exp_repo
        self.decay = decay
        self.reward_proc = reward_proc
        self.updates_to_refresh = 100
        self.updates_since_refresh = 0


        if self.parent is not None:
            self.graph = apply_rule(parent.graph, source_action, grammar)
        else:
            self.graph = None


        if not graph_is_terminal(self.graph):
            self.log_action_probs = exp_repo.get_log_probs_for_graph(self.graph).reshape((-1,))


            self.children = [None for _ in range(len(grammar)*
                                                 (1 if self.graph is None else len(self.graph)))] # maps action index to child


            log_freqs = get_log_conditional_freqs(grammar, [self.graph])
            # the full_logit_priors at this stage merely store the information about which actions are allowed
            full_logit_priors, _, node_mask = get_full_logit_priors(self.grammar, self.max_depth - self.depth,
                                                                    [self.graph])  # TODO: proper arguments
            priors = full_logit_priors + log_freqs
            self.log_priors = priors.reshape((-1,))
            self.probs_from_log_action_probs()

            self.result_repo = RuleChoiceRepository(len(self.log_priors),
                                                    reward_proc=reward_proc,
                                                    mask=self.log_priors > -1e4,
                                                    decay=decay)

    def is_terminal(self):
        return graph_is_terminal(self.graph)

    def value_distr(self):
        return self.value_distr_total/self.child_run_count

    def action_probabilities(self):
        if self.updates_since_refresh > self.updates_to_refresh:
            self.refresh_probabilities()
        self.updates_since_refresh +=1
        return self.probs

    def probs_from_log_action_probs(self):
        total_prob = self.log_action_probs + self.log_priors
        assert total_prob.max() > -1e4, "We need at least one allowed action!"
        tmp = np.exp(total_prob - total_prob.max())
        self.probs = tmp / tmp.sum()
        assert not np.isnan(self.probs.sum())

    # specific
    def refresh_probabilities(self):
        self.log_action_probs = self.result_repo.get_conditional_log_probs()
        self.probs_from_log_action_probs()
        self.updates_since_refresh = 0



    def apply_action(self, action):
        if self.children[action] is None:
            self.make_child(action)
        chosen_child = self.children[action]
        reward, info = get_reward(chosen_child.graph)
        return chosen_child, reward, info

    def make_child(self, action):
        self.children[action] = MCTSNode(grammar=self.grammar,
                                         parent=self,
                                         source_action=action,
                                         max_depth=self.max_depth,
                                         depth=self.depth+1,
                                         exp_repo=self.experience_repository,
                                         decay=self.decay,
                                         reward_proc=self.reward_proc)

    def back_up(self, reward):
        # self.process_back_up(reward)
        if self.parent is not None:
            # update the local result cache
            self.parent.result_repo.update(self.source_action, reward)
            # update the global result cache
            self.experience_repository.update(self.parent.graph, self.source_action, reward)
            self.parent.back_up(reward)

    # # this one is specific
    # def process_back_up(self, reward):
    #     binned_reward = to_bins(reward, num_bins)
    #     if self.value_distr_total is None:
    #         self.value_distr_total = binned_reward
    #         self.child_run_count = 1
    #     else:
    #         self.value_distr_total *= self.decay
    #         self.value_distr_total += binned_reward
    #
    #         self.child_run_count *= self.decay
    #         self.child_run_count += 1


def to_bins(reward, num_bins): # TODO: replace with a ProbabilityDistribution object
    reward = max(min(reward,1.0), 0.0)
    out = np.zeros([num_bins])
    ind = math.floor(reward*num_bins*0.9999)
    out[ind] = 1
    return out


def log_thompson_probabilities(ps):
    """
    Calculate thompson probabilities for one slice, that's already been masked
    :param ps: actions x bins floats reward probability distribution per action
    :return: actions floats
    """
    ps += 1e-5 # to guarantee positivity
    ps = ps/ps.sum(axis=1, keepdims=True)
    log_ps = np.log(ps)
    log_cdfs = np.log(ps.cumsum(axis=1))
    log_cdf_prod = np.sum(log_cdfs, axis=0, keepdims=True)
    pre_thompson = (log_ps[:, 1:] + log_cdf_prod[:, :-1] - log_cdfs[:, :-1])
    thompson = (np.exp(pre_thompson - pre_thompson.max())).sum(1)
    log_thompson = np.log(thompson + 1e-8)
    # thompson2 = (ps*cdf_prod / cdfs).sum(1) # this should always be >1
    total = thompson.sum()
    assert total > 0, "total probs must be positive!"
    out = thompson/total
    assert not np.isnan(log_thompson.sum()), "Something went wrong in thompson probs, got a nan"
    return log_thompson, out

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

if __name__=='__main__':
    num_bins = 50  # TODO: replace with a Value Distribution object
    ver = 'trivial'
    obj_num = 0
    reward_fun = guacamol_goal_scoring_functions(ver)[obj_num]
    grammar_cache = 'hyper_grammar_guac_10k_with_clique_collapse.pickle'  # 'hyper_grammar.pickle'
    grammar_name = 'hypergraph:' + grammar_cache
    max_seq_length = 30
    num_steps = 1
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
                         decay=0.99,
                         reward_proc=lambda x: (1,to_bins(x, num_bins)),
                         refresh_prob_thresh=0.01)
    for _ in range(num_steps):
        rewards, infos = explore(root_node, 100)
        # visualisation code goes here

    print(root_node.result_repo.avg_reward())
    print("done!")