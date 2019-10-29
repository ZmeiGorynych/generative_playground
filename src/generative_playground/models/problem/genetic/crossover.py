from collections import defaultdict
import numpy as np
import random
import torch

# TODO: does this belong in the grammar object?
def ind_to_nonterminal_string(model, ind):
    condition = model.ind_to_condition[ind]
    nonterm = model.grammar.condition_pair_to_nonterminal_string(condition)
    return nonterm


def classic_mutate(model, delta_scale=1, parameter_capper=lambda x: x):
    delta = np.random.normal(scale=delta_scale, size=model.params.shape)
    model.params = parameter_capper(model.params + delta)
    return model


# Question: should we include the priors into picking the top probs?
def mutate(model, pick_best=20):
    print('mutating!')
    internal_model = model.model.stepper.model
    internal_model.collapse_unconditionals()
    total_probs = internal_model.conditionals.data + \
                  internal_model.unconditionals.view(1,-1)
    max_probs = total_probs.max(dim=0)[0]
    best = torch.argsort(max_probs, dim=0, descending=True)[:pick_best]
    nonterminals = [ind_to_nonterminal_string(internal_model, ind.item()) for ind in best]
    # nonterminals = [internal_model.grammar.condition_pair_to_nonterminal_string(pair) for pair in conditions]
    nt_map = defaultdict(list)
    for nti, nt in enumerate(nonterminals):
        nt_map[nt].append(nti)

    # select the indices we want to swap. Must check that they have matching parent nonterminals for the rules to be valid
    nt_map_at_least_two = {nt:inds for nt, inds in nt_map.items() if len(inds) >=2}
    nts = list(nt_map_at_least_two.keys())
    probs = np.array([len(v) for v in nt_map_at_least_two.values()])
    probs = probs/probs.sum()
    chosen_nt = nts[np.random.multinomial(1, probs)[0]]
    chosen_inds = random.sample(nt_map_at_least_two[chosen_nt], 2)
    orig_inds = [best[i] for i in chosen_inds]
    assert ind_to_nonterminal_string(internal_model, orig_inds[0].item()) == \
           ind_to_nonterminal_string(internal_model, orig_inds[1].item())

    # we've merged the unconditional probabilities into the conditionals, now just need to swap the conditional ones
    cond_slice = internal_model.conditionals.data[:, orig_inds[0]].clone()
    internal_model.conditionals.data[:, orig_inds[0]] = internal_model.conditionals.data[:, orig_inds[1]]
    internal_model.conditionals.data[:, orig_inds[1]] = cond_slice
    return model


def classic_crossover(model1, model2, d=0.25, parameter_capper=lambda x: x):
    # to save RAM, we make the child out of model1
    alpha = np.random.uniform(-d, 1+d)
    child = model1
    child.params = model1.params + alpha*(model2.params - model1.params)
    child.params = parameter_capper(child.params)
    return child


def crossover(model1, model2):
    # first collapse the unconditional vectors into the conditionals for cleaner crossover
    model1.model.stepper.model.collapse_unconditionals()
    model2.model.stepper.model.collapse_unconditionals()

    print('crossover!')
    cond2 = model2.model.stepper.model.conditionals
    cmask = torch.FloatTensor(len(cond2)).uniform_(0, 1) > 0.5
    model1.model.stepper.model.conditionals.data[cmask] = cond2.data[cmask]

    return model1
