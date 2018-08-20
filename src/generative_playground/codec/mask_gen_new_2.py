import numpy as np
import nltk
from nltk.grammar import Nonterminal, is_nonterminal, is_terminal
import copy
from frozendict import frozendict

class GrammarMaskGeneratorNew:
    def __init__(self, MAX_LEN, grammar):
        self.MAX_LEN = MAX_LEN
        self.grammar = grammar
        self.do_terminal_mask = True
        self.reset()
        self.term_dist_calc = TerminalDistanceCalculator(self)

    def reset(self):
        self.S = None
        self.t = 0
        self.ring_num_map = {}

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

        for i, a in enumerate(last_actions):
            mask[i, :] = self.process_one_action(i, a)

        self.t += 1
        self.prev_actions = last_actions
        self.mask = mask
        return mask

    def process_one_action(self, i, a):
        if a is not None:
            # 1. Apply the expansion from last prod rule
            this_rule = self.grammar.GCFG.productions()[a]
            # find the token to apply the expansion to
            for this_index, old_token in enumerate(self.S[i]):
                if not isinstance(old_token['token'], str):
                    break
            if this_rule.lhs() != Nonterminal('Nothing'):
                new_tokens = apply_rule(old_token, this_rule, self.t)
                # do the replacement
                self.S[i] = self.S[i][:this_index] + new_tokens + self.S[i][this_index + 1:]

            # 2. generate masks for next prod rule
            # find the index of the next token to expand, which is the first nonterminal in sequence
        for this_index, this_token in enumerate(self.S[i]):
            if not isinstance(this_token['token'], str):
                break
            this_token = {'token': nltk.grammar.Nonterminal('Nothing')}

        # get the formal grammar mask
        self.grammar_mask = self.get_grammar_mask(this_token)

        if this_token['token'] == nltk.grammar.Nonterminal('Nothing'):
            # we only get to this point if the sequence is fully expanded
            return self.grammar_mask


        # get the terminal distance mask
        if self.do_terminal_mask:
            term_distance = sum([self.term_dist_calc(x) for x in self.S[i]])
            steps_left = self.MAX_LEN - self.t - 1
            self.terminal_mask = np.zeros_like(self.grammar_mask)
            rule_dist = self.term_dist_calc.rule_d_term_dist(this_token)
            new_term_dist = rule_dist + term_distance
            self.terminal_mask[new_term_dist < steps_left - 1] = 1
        else:
            self.terminal_mask = np.ones_like(self.grammar_mask)

        # if we're expanding a ring numeric token
        self.ring_mask = self.get_ring_mask(this_token, i, this_index)

        mask = self.grammar_mask * self.terminal_mask * self.ring_mask
        [p for ip, p in enumerate(self.grammar.GCFG.productions()) if self.terminal_mask[ip]]
        assert(not all([x == 0 for x in mask]))
        return mask

    def get_ring_mask(self, this_token, i=None, this_index=None):
        if 'ringID' in this_token:
            # if it's a numeral, choose which one to use
            if str(this_token['token']._symbol[:3]) == 'num':
                full_num_ID = (i, this_token['ringID'], this_token['token']._symbol)
                if i is None: # this happens when we call this for terminal distance cald
                    nums_to_use = self.grammar.numeric_tokens
                elif full_num_ID in self.ring_num_map:
                    # this is the closing numeral of a ring
                    nums_to_use = [self.ring_num_map.pop(full_num_ID)]
                else:
                    # this is the opening numeral of a ring
                    # go through the sequence up to this point, collecting all tokens occurring an odd number of times
                    used_tokens = set()
                    for j in range(this_index):
                        current_token = self.S[i][j]['token']
                        if current_token in self.grammar.numeric_tokens:
                            if current_token in used_tokens:
                                used_tokens.remove(current_token)
                            else:
                                used_tokens.add(current_token)

                    # find the first unused numeric token
                    for nt in self.grammar.numeric_tokens:
                        if nt not in used_tokens:
                            break
                    # that is the num we want to use and store for later
                    nums_to_use = [nt]
                    self.ring_num_map[full_num_ID] = nt

                ring_mask = np.array(
                    [1 if a.rhs()[0] in nums_to_use else 0 for a in self.grammar.GCFG.productions()])
            else:
                ring_mask = np.ones([len(self.grammar.GCFG.productions())])
                if this_token['ring_size'] > 6: # 4# max cycle length 6
                    # block out cycle continuation
                    for k in self.grammar.cycle_continues:
                        ring_mask[k] = 0
                if this_token['ring_size'] < 2: #4# minimum cycle length 5
                    # block out cycle finish
                    for k in self.grammar.cycle_ends:
                        ring_mask[k] = 0
        else:
            ring_mask = np.ones([len(self.grammar.GCFG.productions())])
        return ring_mask

    def get_grammar_mask(self, token):
        try:
            token = token['token'] # in case we got a dict
        except:
            pass
        grammar_mask = self.grammar.masks[self.grammar.lhs_to_index[token], :]
        return grammar_mask

class TerminalDistanceCalculator:
    def __init__(self, mask_gen):
        self.mask_gen = copy.copy(mask_gen)
        self.mask_gen.do_terminal_mask = False
        self.term_dist = {}
        self.GCFG = mask_gen.grammar.GCFG

        for p in self.GCFG.productions():
            #self.term_dist[frozendict({'token': p.lhs()})] = float('inf')
            for s in p.rhs():
                if is_terminal(s):
                    # terminals have term distance 0
                    self.term_dist[frozendict({'token': s})] = 0

        # seed the search with the root symbol
        self.term_dist[frozendict({'token': Nonterminal('smiles')})] = float('inf')
        self.term_dist[frozendict({'token': Nonterminal('None')})] = 0
        #self.term_dist[frozendict({'token': Nonterminal('forbidden_token')})] = float('inf')

        last_term_dist ={}
        while True: # iterate to convergence
            last_term_dist = copy.copy(self.term_dist)
            for sym in last_term_dist.keys():
                if self.term_dist[sym] > 0:
                    mask = self.get_mask_from_token(sym)
                    # [p for ip, p in enumerate(self.GCFG.productions()) if mask[ip]]
                    assert (not all([x == 0 for x in mask]))
                    for ip, p in enumerate(self.GCFG.productions()):
                        if mask[ip]:
                            this_exp = apply_rule(sym, p, 0)
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
                                print(p, self.term_dist[frozendict(sym)], this_term_dist,
                                      [self.term_dist[frozendict(this_sym)] for this_sym in this_exp])
                                self.term_dist[frozendict(sym)] = this_term_dist


            if last_term_dist == self.term_dist:
                break

    def __call__(self, sym):
        return self.term_dist[frozendict({k: (0 if k == 'ringID' else v) for k,v in sym.items()})]

    def get_mask_from_token(self, sym):
        grammar_mask = self.mask_gen.get_grammar_mask(sym)
        ring_mask = self.mask_gen.get_ring_mask(sym)
        mask = grammar_mask*ring_mask
        assert(any(mask))
        return mask


    def rule_d_term_dist(self, x):
        # calculates the change in terminal distance by each of the potential rules starting from x
        d_td = []
        mask = self.get_mask_from_token(x)
        for ip, p in enumerate(self.GCFG.productions()):
            # TODO: fall back to lookup where that is possible
            if mask[ip] and p.lhs() == x['token']:
                new_tokens = apply_rule(x, p, 0)
                diff_td = sum([self.__call__(n) for n in new_tokens]) - self.__call__(x)
                # that's the definition of term distance, SOME rule reduces it by one

                d_td.append(diff_td)
            else:
                d_td.append(0)

        d_td = np.array(d_td)
        assert(any(d_td == -1))
        return d_td


def apply_rule(this_token, this_rule, t):
    this_inner_token = this_token['token']
    # do some safety checks
    assert (this_inner_token == this_rule.lhs())
    if ('cycle' in this_inner_token._symbol or 'num' in this_inner_token._symbol) \
        and 'ring_size' not in this_token:
        # 'cycle' and 'num' tokens only appear in cycles, where they are assigned ring_sizes
        raise ValueError("'cycle' and 'num' tokens only appear in cycles, where they are assigned ring_sizes")
    # get the expansion
    new_tokens = [{'token': x} for x in this_rule.rhs()]

    # if the expansion is a new ring, give it a ringID=t
    if 'ring' in this_token['token']._symbol:
        ringID = t
        ring_size = 1

    # if the token to be replaced already has a ringID, propagate it
    elif 'ringID' in this_token:
        ringID = this_token['ringID']
        ring_size = this_token['ring_size'] + 1
    else:
        ringID = None
    propagate_strings = ['cycle', 'num', 'starting', 'final']
    if ringID is not None:
        for x in new_tokens:
            x_ = x['token']
            if is_nonterminal(x_):
                if any([ps in x_._symbol for ps in propagate_strings]):
                    x['ringID'] = ringID
                    x['ring_size'] = ring_size

    return new_tokens

def terminal_distance(grammar, x):
    # due to masking that enforces minimal ring length, must override term distances derived purely from grammar
    if x['token'] == Nonterminal('aliphatic_ring'):
        return 8
    elif x['token'] == Nonterminal('cycle_bond'):
        return max(2, 7 - x['ring_size'])
    elif x['token'] == Nonterminal('cycle_double_bond'):
        # need to go at least to cycle_bond -> num1 -> number
        return max(3, 7 - x['ring_size'])
    else:
        return grammar.terminal_dist(x['token'])

def rule_d_term_dist(grammar, x, t):
    # calculates the change in terminal distance by each of the potential rules starting from x
    d_td = []
    for p in grammar.GCFG.productions():
        # TODO: fall back to lookup where that is possible
        if p.lhs() == x['token']:
            new_tokens = apply_rule(x, p, t)
            d_td.append(sum([terminal_distance(grammar, n) for n in new_tokens]) -
                        terminal_distance(grammar, x))
        else:
            d_td.append(0)

    return np.array(d_td)