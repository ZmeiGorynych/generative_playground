import numpy as np
import torch.nn.functional as F
from generative_playground.utils.gpu_utils import device
from generative_playground.codec.hypergraph import expand_index_to_id
from generative_playground.codec.hypergraph_mask_generator import get_log_conditional_freqs, get_full_logit_priors, \
    action_to_node_rule_indices, apply_one_action
from generative_playground.models.problem.mcts.result_repo import log_thompson_probabilities, update_node
import torch


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
        if parent is None:
            self.id = tuple()
        else:
            self.id = tuple(list(parent.id)+[source_action])

        global_params.state_store[self.id] = LocalState()

        self.globals = global_params

        self.parent = parent
        self.source_action = source_action
        self.depth = depth

        if self.parent is not None:
            self.locals().graph = apply_rule(self.parent.locals().graph, source_action, global_params.grammar)
        else:
            self.locals().graph = None

        if not self.is_terminal():
            num_children = len(global_params.grammar) * (1 if self.locals().graph is None else len(self.locals().graph))
            self.children = np.empty([num_children], dtype=np.dtype(object))  # maps action index to child
            # the full_logit_priors at this stage merely store the information about which actions are allowed
            full_logit_priors, _, node_mask = get_full_logit_priors(self.globals.grammar,
                                                                    self.globals.max_depth - self.depth,
                                                                    [self.locals().graph])
            # these are the emprirical priors from the original dataset
            log_freqs = get_log_conditional_freqs(self.globals.grammar, [self.locals().graph])

            self.locals().log_priors = (full_logit_priors + log_freqs).reshape((-1,))

    def locals(self):
        return self.globals.state_store[self.id]

    def is_terminal(self):
        return graph_is_terminal(self.locals().graph)

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

    def update(self, reward): #it makes sense to call this as part of back up, perhaps
        raise NotImplementedError

    def back_up(self, reward):
        raise NotImplementedError

    def action_probabilities(self):
        raise NotImplementedError


class MCTSModelParent(MCTSNodeParent):
    def apply_action(self, action):
        chosen_child, reward, info = super().apply_action(action)
        self.used_logp = self.logp[action]
        self.logp = None
        return chosen_child, reward, info

    def back_up(self, reward, log_ps=None):
        if log_ps is None:  # terminal node
            log_ps = []
        else:
            log_ps += [self.used_logp]
            self.used_logp = None

        self.update(reward)

        if self.parent is None:
            self.globals.process_reward(reward, log_ps, self.globals.model.parameters())
        else:
            self.parent.back_up(reward, log_ps)

    def action_probabilities(self):
        locals = self.locals()
        action_logits = self.globals.model([locals.graph], locals.log_priors)['masked_policy_logits'][0]
        if hasattr(locals, 'local_logits'):
            action_logits += locals.local_logits
        action_probs = F.softmax(action_logits)
        self.logp = torch.log(action_probs + 1e-5)
        np_probs = action_probs.cpu().detach().numpy()
        return np_probs

    def parameters(self):
        return NotImplementedError

    def update(self, reward):
        pass

class MCTSNodeGlobalModel(MCTSModelParent):
    def parameters(self):
        return self.globals.model.parameters()


class MCTSNodeLocalModel(MCTSModelParent):
    def __init__(self, global_params, parent, source_action, depth):
        super().__init__(global_params, parent, source_action, depth)
        if not self.is_terminal():
            self.locals().local_logits = torch.zeros(self.locals().log_priors.shape, device=device)

    def parameters(self):
        return self.globals.model.parameters() + [self.locals().local_logits]

class MCTSNodeGlobalModelLocalThompson(MCTSNodeGlobalModel):
    def __init__(self, global_params, parent, source_action, depth):
        super().__init__(global_params, parent, source_action, depth)
        self.updates_since_refresh = 0
        self.locals().total_weight = 0.0
        self.locals().total_reward = 0.0

        if not self.is_terminal():
            mask = self.locals().log_priors.reshape([-1]) > -1e4
            self.locals().mask = mask
            self.locals().mask_len = len(mask[mask])
            self.locals().local_logits = torch.zeros(self.locals().log_priors.shape,
                                                     device=device,
                                                     requires_grad=False)# we'll update them using Thompson probabilities

    def experiences(self):
        return self.locals().total_weight, self.locals().total_reward

    def update(self, reward):
        self.locals().total_weight, \
        self.locals().total_reward = update_node(self.locals().total_weight,
                                                           self.locals().total_reward,
                                                           reward,
                                                           self.globals.decay,
                                                           self.globals.reward_proc)
        self.updates_since_refresh +=1
        if self.updates_since_refresh > self.globals.updates_to_refresh:
            self.update_local_logits()

    # this flavor uses a model globally, but adds thompson-generated local logits in each node
    def update_local_logits(self):
        # already the pre-filtered
        my_total_reward = np.zeros((self.locals().mask_len,
                                    self.globals.reward_proc.num_bins))
        my_total_wt = np.zeros(self.locals().mask_len)
        count = 0
        for ind, this_mask in enumerate(self.locals().mask):
            if this_mask:
                if self.children[ind] is not None:
                    my_total_wt[count], my_total_reward[count] = self.children[ind].experiences()
                count += 1

        max_wt = 1/(1-self.globals.decay)

        # Get my own average reward distribution, this is a fallback if we don't have enough observations
        my_avg_wt = np.maximum(0.0, max_wt - my_total_wt)
        avg_wt, avg_reward = self.experiences()
        avg_reward /= avg_wt

        regularized_rewards = my_total_reward + avg_reward[None,:]*my_avg_wt[:, None]
        regularized_rewards /= my_total_wt[:, None] + my_avg_wt[:, None]

        log_ps, ps = log_thompson_probabilities(regularized_rewards)
        count = 0
        for i, m in enumerate(self.locals().mask):
            if m:
                self.locals().local_logits[i] = log_ps[count]
                count += 1
        self.updates_since_refresh = 0

class MCTSThompsonParent(MCTSNodeParent):
    def __init__(self,
                 global_params,
                 parent,
                 source_action,
                 depth):
        """
        Creates a placeholder with just the value distribution guess, to be aggregated by the parent
        :param value_distr:

        """
        super().__init__(global_params, parent, source_action, depth)

        self.updates_since_refresh = 0
        self.locals().total_weight = 0.0
        self.locals().total_reward = 0.0

        if not self.is_terminal():
            mask = self.locals().log_priors.reshape([-1]) > -1e4
            self.locals().mask = mask
            self.locals().mask_len = len(mask[mask])
            # self.locals().result_repo = self.globals.rule_choice_repo_factory(mask)

            self.locals().log_action_probs = None
            self.locals().probs = None

            self.init_action_probabilities()
        assert hasattr(self.locals(), 'graph')

    def action_probabilities(self):
        if self.updates_since_refresh > self.globals.updates_to_refresh:
            self.refresh_probabilities()
        self.updates_since_refresh += 1
        return self.locals().probs

    def back_up(self, reward):
        # self.process_back_up(reward)
        if self.parent is not None:
            # update the local result cache
            self.update(reward)
            # update the global result cache
            self.globals.experience_repository.update(self.parent.locals().graph, self.source_action, reward)
            # and recurse
            self.parent.back_up(reward)

    def update(self, reward):
        self.locals().total_weight, \
        self.locals().total_reward = update_node(self.locals().total_weight,
                                                           self.locals().total_reward,
                                                           reward,
                                                           self.globals.decay,
                                                           self.globals.reward_proc)

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

    def experiences(self):
        return self.locals().total_weight, self.locals().total_reward

class MCTSNodeLocalThompson(MCTSThompsonParent):
    def init_action_probabilities(self):
        # gets called on node creation
        self.locals().log_action_probs = -1e5*np.zeros(len(self.locals().mask))
        # self.locals().log_action_probs = self.globals.experience_repository.get_log_probs_for_graph(self.locals().graph).reshape((-1,))
        self.refresh_probabilities()

    def refresh_probabilities_old(self):
        # gets called each time a node decides to refresh its probability vector
        self.locals().log_action_probs = self.locals().result_repo.get_conditional_log_probs()
        self.generate_probs_from_log_action_probs()
        self.updates_since_refresh = 0

    def refresh_probabilities(self):
        my_mask = self.locals().mask
        # already the pre-filtered
        glob_wt, glob_rewards = self.globals.experience_repository.get_total_rewards_for_graph(self.locals().graph, my_mask)
        rule_wt, rule_rewards = self.globals.experience_repository.get_total_rewards_by_rule_for_graph(self.locals().graph, my_mask)
        # my own cache stores experiences by (graph node, rule) combinations directly

        my_reward = np.zeros((self.locals().mask_len, self.globals.num_bins))
        my_wt = np.zeros(self.locals().mask_len)
        count = 0
        for ind, this_mask in enumerate(my_mask):
            if this_mask:
                if self.children[ind] is not None:
                    my_wt[count], my_reward[count] = self.children[ind].experiences()
                count += 1

        max_wt = 1/(1-self.globals.decay)

        # TODO: cast the below as a for loop
        my_glob_wt = np.maximum(0.0, max_wt - my_wt)
        my_rule_wt = np.maximum(0.0, max_wt - my_wt - my_glob_wt)
        my_avg_wt = np.maximum(0.0, max_wt - my_wt - my_glob_wt-my_rule_wt)
        avg_reward = rule_rewards.mean(axis=0, keepdims=True)

        regularized_rewards = my_reward*my_wt[:, None] + glob_rewards*my_glob_wt[:,None] \
                              + rule_rewards*rule_wt[:, None] + avg_reward*my_avg_wt[:, None]
        regularized_rewards /= my_wt[:, None] + my_glob_wt[:,None] + rule_wt[:, None] + my_avg_wt[:,None]

        log_ps, ps = log_thompson_probabilities(regularized_rewards)
        self.locals().log_action_probs[my_mask] = 0.25*log_ps

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
        return reward_fun(smiles), {'smiles': smiles}
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
