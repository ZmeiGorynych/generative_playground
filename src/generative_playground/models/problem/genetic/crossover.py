from collections import defaultdict
import numpy as np
import random
import torch

def ind_to_nonterminal_string(model, ind):
    condition = model.ind_to_condition[ind]
    nonterm = model.grammar.condition_pair_to_nonterminal_string(condition)
    return nonterm

# Question: should we include the priors into picking the top probs?
def mutate(model, pick_best=20):
    print('mutating!')
    internal_model = model.model.stepper.model
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

     # now swap the two rows
    uncond_slice = internal_model.unconditionals.data[orig_inds[0]].clone()
    internal_model.unconditionals.data[orig_inds[0]] = internal_model.unconditionals.data[orig_inds[1]]
    internal_model.unconditionals.data[orig_inds[1]] = uncond_slice

    cond_slice = internal_model.conditionals.data[:, orig_inds[0]].clone()
    internal_model.conditionals.data[:, orig_inds[0]] = internal_model.conditionals.data[:, orig_inds[1]]
    internal_model.conditionals.data[:, orig_inds[1]] = cond_slice
    return model


def crossover(model1, model2):
    print('crossover!')
    cond2 = model2.model.stepper.model.conditionals
    cmask = torch.FloatTensor(len(cond2)).uniform_(0, 1) > 0.5
    model1.model.stepper.model.conditionals.data[cmask] = cond2.data[cmask]

    uncond2 = model2.model.stepper.model.unconditionals
    umask = torch.FloatTensor(len(uncond2)).uniform_(0, 1) > 0.5
    model1.model.stepper.model.unconditionals.data[umask] = uncond2.data[umask]

    return model1