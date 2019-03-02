import numpy as np
import uuid
import nltk
from nltk.grammar import Nonterminal, is_nonterminal, is_terminal
import copy
from frozendict import frozendict
from generative_playground.codec.grammar_helper import grammar_zinc_new


def get_grammar_mask(token, grammar):
    try:
        token = token['token'] # in case we got a dict
    except:
        pass

    grammar_mask = grammar.masks[grammar.lhs_to_index[token], :]
    return grammar_mask

def get_ring_mask(this_token, grammar, ring_num_map=None, this_S=None, this_index=None):
    assert(this_S is None or this_token == this_S[this_index])
    if 'num' in this_token: # if this token is part of a cycle
        # if it's a numeral, choose which one to use from its stored guid
        if str(this_token['token']._symbol) in ['num', 'num1']:
            if this_token['num'] is None:
                nums_to_use = grammar.numeric_tokens
            elif this_token['num'] in ring_num_map: # if we're closing a cycle
                nums_to_use = [ring_num_map[this_token['num']]]
            else: # if we're opening a cycle
                free_numerals = find_free_numerals(this_S,
                                                   this_index=this_index,
                                                   grammar=grammar)
                nums_to_use = [free_numerals[0]]
                ring_num_map[this_token['num']] = free_numerals[0]

            ring_mask = np.array(
                [1 if a.rhs()[0] in nums_to_use else 0 for a in grammar.GCFG.productions()])
        else: # control minimal and maximal ring size
            ring_mask = np.ones([len(grammar.GCFG.productions())])
            if this_token['size'] > 6: # 4# max cycle length 6
                # block out cycle continuation
                for k in grammar.cycle_continues:
                    ring_mask[k] = 0
            if this_token['size'] < 4: #4# minimum cycle length 5
                # block out cycle finish
                for k in grammar.cycle_ends:
                    ring_mask[k] = 0
    else:
        ring_mask = np.ones([len(grammar.GCFG.productions())])
    return ring_mask

def apply_rule(S, this_index, this_rule, grammar, checks=True):
    this_token = dict(S[this_index])
    this_inner_token = this_token['token']
    # do some safety checks
    if checks:
        assert (this_inner_token == this_rule.lhs())
        if ('cycle' in this_inner_token._symbol or 'num' in this_inner_token._symbol) \
            and 'size' not in this_token:
            # 'cycle' and 'num' tokens only appear in cycles, where they are assigned ring_sizes
            raise ValueError("'cycle' and 'num' tokens only appear in cycles, where they are assigned ring_sizes")

    # get the expansion
    new_tokens = [{'token': x} for x in this_rule.rhs()]

    propagate_strings = ['cycle', 'num']
    num_nonterms = ['num', 'num1']
    # if the expansion is a new ring, assign the numeral to use
    num_map ={}
    if 'ring' in this_token['token']._symbol:
        num_id = uuid.uuid4()
        num1_id = uuid.uuid4()
        for x in new_tokens:
            if is_nonterminal(x['token']) and \
                    any([ps in x['token']._symbol for ps in propagate_strings]):
                x['size'] = 1
                if grammar is not None:
                    if x['token'] == Nonterminal('num1'): # this is very hacky, to do better want modular aromatic cycles
                        x['num'] = num1_id
                    else:
                        x['num'] = num_id
                else:
                    x['num'] = None
        else:
            this_token['num'] = None
        this_token['size'] = 0

    elif 'num' in this_token:
        if this_token['token']._symbol in ['num', 'num1']:
            # tag the resulting terminal so we know it's a cycle numeral, not a charge numeral
            for x in new_tokens:
                x['is_cycle_numeral'] = True
        # if this_token is a cycle propagation token, propagate the numeral and size counter
        for x in new_tokens:
            if is_nonterminal(x['token']) and \
                    any([ps in x['token']._symbol for ps in propagate_strings]):
                x['num'] = this_token['num']
                x['size'] = this_token['size'] + rule_adds_atom(this_rule)

    if checks:
        for x in new_tokens:
            if is_nonterminal(x['token']) and \
                    any([ps in x['token']._symbol for ps in propagate_strings]):
                assert('num' in x and 'size' in x)

    for x in new_tokens:
        try:
            x['term_dist'] = term_dist_calc(x)
        except:
            pass

    return new_tokens

def rule_adds_atom(p):
    atoms = ['c', 'n', 'o', 's', 'f', 'cl', 'br', 'i']
    if any([x.lower() in atoms for x in p.rhs() if is_terminal(x)]) or \
        any(['valence' in x._symbol for x in p.rhs() if is_nonterminal(x)]):
        return 1
    elif any(['segment' in x._symbol for x in p.rhs() if is_nonterminal(x)]):
        return 2
    else:
        return 0

class TerminalDistanceCalculator:
    def __init__(self, grammar=grammar_zinc_new, checks=False):
        # self.mask_gen = get_mask_gen()
        # self.mask_gen.do_terminal_mask = False
        self.term_dist = {}
        self.d_term_dist = {}
        self.grammar = grammar
        self.GCFG = self.grammar.GCFG
        self.checks = checks

        for p in self.GCFG.productions():
            for s in p.rhs():
                if is_terminal(s):
                    # terminals have term distance 0
                    self.term_dist[frozendict({'token': s})] = 0

        self.term_dist[frozendict({'token': Nonterminal('None')})] = 0

        # seed the search with the root symbol
        self.term_dist[frozendict({'token': Nonterminal('smiles')})] = float('inf')

        while True: # iterate to convergence
            # print('*** and one more pass... ***')
            last_term_dist = copy.copy(self.term_dist)
            for sym in last_term_dist.keys():
                if is_terminal(sym['token']):
                    self.term_dist[sym] = 0
                if self.term_dist[sym] > 0:
                    mask = self.get_mask_from_token(sym)
                    # [p for ip, p in enumerate(self.GCFG.productions()) if mask[ip]]
                    if self.checks:
                        assert (not all([x == 0 for x in mask]))
                    for ip, p in enumerate(self.GCFG.productions()):
                        if mask[ip]:
                            # print('trying', sym, p)
                            this_exp = apply_rule([sym], 0, p, None, self.checks)
                            this_term_dist = 1
                            for this_sym in this_exp:
                                if frozendict(this_sym) not in self.term_dist:
                                    self.term_dist[frozendict(this_sym)] = float('inf')
                                    print('added ', this_sym, 'from', sym, 'via', p)
                                    # if 'ring_size' in sym and sym['ring_size'] > 6:
                                    #     print('aaa')
                                this_term_dist += self.term_dist[frozendict(this_sym)]
                            if this_term_dist < self.term_dist[frozendict(sym)]:
                                # if 'ring_size' in sym and sym['ring_size'] > 6:
                                #     print('aaa')
                                print('improving:', p, self.term_dist[frozendict(sym)], this_term_dist,
                                      [self.term_dist[frozendict(this_sym)] for this_sym in this_exp])
                                self.term_dist[frozendict(sym)] = this_term_dist

            if last_term_dist == self.term_dist:
                break

    def __call__(self, sym):
        my_key = token_to_hashable(sym)
        try:
            return self.term_dist[my_key]
        except:
            return float('1e6')

    def get_mask_from_token(self, sym):
        grammar_mask = get_grammar_mask(sym, self.grammar)
        ring_mask = get_ring_mask(sym, self.grammar)
        mask = grammar_mask*ring_mask
        # if is_nonterminal(sym['token']) and sym['token']._symbol == 'nonH_bond':
        #     print('let''s see...')
        if self.checks:
            assert any(mask)
        return mask


    def rule_d_term_dist(self, x):
        # calculates the change in terminal distance by each of the potential rules starting from x
        d_td = []
        x = token_to_hashable(x)
        mask = self.get_mask_from_token(x)
        if x not in self.d_term_dist:
            # calculate d_term_dist for that extended token
            for ip, p in enumerate(self.GCFG.productions()):
                if mask[ip] and p.lhs() == x['token']:
                    new_tokens = apply_rule([x], 0, p, None, self.checks)
                    diff_td = sum([self.__call__(n) for n in new_tokens]) - self.__call__(x)

                    d_td.append(diff_td)
                else:
                    d_td.append(0)

            d_td = np.array(d_td)
            if self.checks:
                # that's the definition of term distance, SOME rule reduces it by one, but never by more
                assert(np.min(d_td) == -1)
            self.d_term_dist[x] = d_td

        return self.d_term_dist[x]

term_dist_calc = TerminalDistanceCalculator()

class GrammarMaskGeneratorNew:
    def __init__(self, MAX_LEN, grammar, sanity_checks=True):
        self.MAX_LEN = MAX_LEN
        self.grammar = grammar
        self.do_terminal_mask = True
        self.S = None
        self.Stree = None
        self.t = 0
        self.ring_num_map = {}
        self.roots = []
        self.reset()
        self.checks = sanity_checks
        self.term_dist_calc = term_dist_calc

    def reset(self):
        self.S = None
        self.Stree = None
        self.t = 0
        self.ring_num_map = {}
        self.roots = []

    def __call__(self, last_actions):
        """
        Returns the 'smart' mask
        :param last_actions:
        :return:
        """
        if self.t >= self.MAX_LEN:
            raise StopIteration("maximum sequence length exceeded for decoder")

        mask = np.zeros([len(last_actions), len(self.grammar.GCFG.productions())])

        if self.S is None:
            # populate the sequences with the root symbol
            self.S = [[{'token': Nonterminal('smiles')}] for _ in range(len(last_actions))]
            for s in self.S:
                s[0]['term_dist'] = self.term_dist_calc(s[0])
            self.Stree =[[x for x in y] for y in self.S]

        for i, a in enumerate(last_actions):
            self.S[i], mask[i, :] = self.process_one_action(self.S[i], a)

        self.t += 1
        self.prev_actions = last_actions
        self.mask = mask
        return mask


    def process_one_action(self, this_S, a):
        if a is not None:
            # 1. Apply the expansion from last prod rule
            this_rule = self.grammar.GCFG.productions()[a]
            # find the token to apply the expansion to
            for this_index, old_token in enumerate(this_S):
                if is_nonterminal(old_token['token']):
                    break
            if this_rule.lhs() != Nonterminal('Nothing'):
                new_tokens = apply_rule(this_S, this_index, this_rule, self.grammar, self.checks)#apply_rule(old_token, this_rule, self.t)
                # do the replacement
                if self.checks:
                    this_S[this_index]['children'] = new_tokens
                this_S = this_S[:this_index] + new_tokens + this_S[this_index + 1:]

        # 2. generate masks for next prod rule
        # find the index of the next token to expand, which is the first nonterminal in sequence
        for this_index, this_token in enumerate(this_S):
            if is_nonterminal(this_token['token']):
                break
            this_token = {'token': nltk.grammar.Nonterminal('Nothing')}

        # get the formal grammar mask
        self.grammar_mask = self.get_grammar_mask(this_token)

        if this_token['token'] == nltk.grammar.Nonterminal('Nothing'):
            # # we only get to this point if the sequence is fully expanded
            return this_S, self.grammar_mask


        # get the terminal distance mask
        if self.do_terminal_mask:
            term_distance = sum([x['term_dist'] for x in this_S])#sum([self.term_dist_calc(x) for x in this_S])
            steps_left = self.MAX_LEN - self.t - 1
            self.terminal_mask = np.zeros_like(self.grammar_mask)
            rule_dist = self.term_dist_calc.rule_d_term_dist(this_token)
            new_term_dist = rule_dist + term_distance
            self.terminal_mask[new_term_dist < steps_left - 1] = 1
        else:
            self.terminal_mask = np.ones_like(self.grammar_mask)

        # if we're expanding a ring numeric token
        self.ring_mask = self.get_ring_mask(this_token, this_S, this_index)

        mask = self.grammar_mask * self.terminal_mask * self.ring_mask

        if self.checks:
            assert(not all([x == 0 for x in mask]))
        return this_S, mask

    def get_ring_mask(self, this_token, this_S=None, this_index=None):
        return get_ring_mask(this_token, self.grammar, self.ring_num_map, this_S, this_index)

    def get_grammar_mask(self, token):
        return get_grammar_mask(token, self.grammar)


def token_to_hashable(x):
    return frozendict({k: (None if k == 'num' else v) for k,v in x.items() if k != 'term_dist'})


def find_free_numerals(S, this_index, grammar, reuse_numerals=True):
    # collect all the un-paired numeral terminals before current token
    used_tokens = set()
    for j in range(this_index):  # up to, and excluding, this_index
        current_token = S[j]['token']
        assert(is_terminal(current_token)) # we assume the token we want to expand now is the leftmost nontermonal
        # the second check is to exclude numerals that describe charge
        if current_token in grammar.numeric_tokens and 'is_cycle_numeral' in S[j]:
            if current_token in used_tokens and reuse_numerals: #this cycle has been closed, can reuse the numeral
                used_tokens.remove(current_token)
            else:
                used_tokens.add(current_token)

    # find the first unused numeral
    free_numerals = [nt for nt in grammar.numeric_tokens if not nt in used_tokens]
    if not free_numerals:
        raise ValueError("Too many nested cycles - can't find a valid numeral")
    else:
        return free_numerals

