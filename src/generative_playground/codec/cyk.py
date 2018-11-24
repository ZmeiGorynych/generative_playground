from nltk.grammar import Nonterminal, Production, is_nonterminal
import numpy as np
import copy

def binarize(p):
    return _binarize(p,[])

def get_unique_num():
    n = 0
    while True:
        yield n
        n+=1

counter = get_unique_num()

def _binarize(p, so_far=[]):
    if len(p.rhs()) <= 2:
        so_far.append(p)
        return so_far
    else:
        new_nont = Nonterminal(p.lhs()._symbol + '_' +str(next(counter)))
        so_far.append(Production(p.lhs(),[p.rhs()[0], new_nont]))
        return _binarize(Production(new_nont, p.rhs()[1:]),so_far)

def get_terminals_nonterminals(prods):
    # preserve order of appearance for nonterminals so we know which token is root
    terminals = set()
    nonterminals = []
    for p in prods:
        if p.lhs() not in nonterminals:
            nonterminals.append(p.lhs())
        terminals = terminals.union(set(p.rhs()))

    terminals = list(terminals.difference(set(nonterminals)))

    return terminals, nonterminals

def cyk_parser(prods, tokenized_seqs):
    # eliminate rules of the shape nonterminal -> nonterminal
    # assume the lhs of first rule is root token
    #non_sing_prods = eliminate_singular_rules(prods, prods[0].lhs())

    # split all rules with len(rhs) > 2
    new_prods = []
    for p in prods:
        new_prods += binarize(p)

    # count and number all terminals
    terminals, nonterminals = get_terminals_nonterminals(new_prods)

    rule_by_lhs = {nt:[] for nt in nonterminals}
    for p in new_prods:
        rule_by_lhs[p.lhs()].append(p)

    rule_cost = {rule: -1 for rule in new_prods + terminals}
    # number all tokens
    # this makes sure the root nonterminal gets index 0
    ind = {val:i for i,val in enumerate(nonterminals + terminals)}

    for seq in tokenized_seqs:
        for t in seq:
            if t not in terminals:
                raise ValueError('unknown token: ' + str(t))

    # create a map from lhs to rhs indices
    seq_len = np.array([len(seq) for seq in tokenized_seqs]).max()

    # run the cyk algorithm, modified to support rules with one item on the rhs
    # meaning of the indices is span-1, start_ind, index of grammar rule
    costs = []
    for tokenized_seq in tokenized_seqs:
        P = -1e6*np.ones([seq_len, seq_len, len(ind)])
        for s, t in enumerate(tokenized_seq):
            P[0, s, ind[t]] = rule_cost[t]

        # go over singular rules at the same level until we know we got them all

        def process_singular_rules(new_prods, P, L):
            changed = True
            while changed:
                changed = False
                for s, t in enumerate(tokenized_seq):
                    for prod in new_prods:
                        if len(prod.rhs()) == 1:
                            old_val = P[L, s, ind[prod.lhs()]]
                            new_val = rule_cost[prod] + P[L, s, ind[prod.rhs()[0]]]
                            if new_val > old_val:
                                P[L, s, ind[prod.lhs()]] = new_val
                                changed = True
            return P

        P = process_singular_rules(new_prods, P, 0)
        print('****:',P[0,:,:].max())
        # for each L = 1 to n-1 -- Length of span -1
        for L in range(1,seq_len):
            #   for each s = 0 to n-L -- Start of span-1
            for s in range(seq_len-L):
                # for each possible lhs
                # process all 'normal' (len(rhs) ==1) rules
                for lhs, prod_list in rule_by_lhs.items():
                    values = [np.zeros([0])]
                    # for all production rules with this lhs
                    for prod in prod_list:
                        #print(prod)
                        # if the rule only has one item on the RHS:
                        if len(prod.rhs()) == 2:
                            tmp = np.zeros([L])
                            for pp in range(L):
                                tmp[pp] = rule_cost[prod] + P[pp, s, ind[prod.rhs()[0]]] \
                                          + P[L - pp - 1, s + pp + 1, ind[prod.rhs()[1]]]
                            values.append(tmp)
                            #print(tmp.max())
                    tmp2 = np.concatenate(values)
                    if len(tmp2) > 0:
                        P[L,s,ind[lhs]] = tmp2.max()

            # and now process all singular rules for this L
            P = process_singular_rules(new_prods, P, L)
            print('*********',L, P[L,:,:].max())
        #           set back[l,s,a] = <p,b,c>

        # convert back into a tree with the original grammar
        costs.append(P[seq_len-1, 0, 0])
    return costs

def is_singular(prod):
    return  len(prod.rhs()) == 1 and is_nonterminal(prod.rhs()[0])

def eliminate_singular_rules(prods, root_token):
    # eliminates all rules whose rhs has only one member, by substitution

    terminals, nonterminals = get_terminals_nonterminals(prods)
    # find all singular rules p with p.lhs = nt
    # check if there are any non-singular rules or whether we can eliminate nt
    singles = {nt: [] for nt in nonterminals}
    others = {nt: [] for nt in nonterminals}
    for prod in prods:
        if is_singular(prod):
            singles[prod.lhs()].append(prod)
        elif len(prod.rhs()) >= 1:
            others[prod.lhs()].append(prod)
        else:
            raise ValueError("rhs must have at least one member! "  + str(prod))

    lhs_has_others = set(others)

    new_prods = prods
    # first replace all singular rules starting with root token:
    new_prods = recursively_replace_root_singulars(new_prods, root_token)

    # for each lhs with singulars:
    for lhs, these_singles in singles.items():
        new_prods = recursively_replace_lhs([p for p in new_prods if p not in these_singles],
                                            lhs, these_singles, lhs in lhs_has_others)
        if len(these_singles) > 0:
            # after we replaced one lhs, need to index the remaining rules all over again
            break

    # check if we still have any singles left, and repeat until done
    if any([len(prod.rhs()) == 1 and is_nonterminal(prod.rhs()[0]) for prod in prods]):
        return eliminate_singular_rules(new_prods, root_token)
    else:
        return new_prods

def recursively_replace_root_singulars(rules, root_token):
    # find a singular root rule
    found = False
    for root_rule in rules:
        if root_rule.lhs() == root_token and is_singular(root_rule):
            found = True
            break

    if not found: # no more root singulars
        return rules

    other_rules = [p for p in rules if p != root_rule]
    new_rules = copy.copy(other_rules)
    for othr in other_rules:
        if othr.lhs() == root_rule.rhs()[0]:
            new_rules = [Production(root_token, othr.rhs())] + new_rules

    return recursively_replace_root_singulars(new_rules, root_token)



def recursively_replace_lhs(rules, lhs, singles, keep_original):
    assert(all([lhs == p.lhs() for p in singles]))
    assert(all([len(p.rhs()) == 1 for p in singles]))
    out = []
    for r in rules:
        if lhs not in r.rhs():
            out += [r]
        else:
            if keep_original:
                out += [r]
            # find first occurrence
            for loc, t in enumerate(r.rhs()):
                if t==lhs:
                    break
            # substitute first occurrence
            new_rules = [Production(r.lhs(),
                                    list(r.rhs()[:loc]) + list(s.rhs()) + list(r.rhs()[loc+1:]))
                         for s in singles]
            out += recursively_replace_lhs(new_rules, lhs, singles, keep_original)

    return out
