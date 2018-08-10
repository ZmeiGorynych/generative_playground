import numpy as np
import nltk
from nltk.grammar import Nonterminal

class GrammarMaskGeneratorNew:
    def __init__(self, MAX_LEN, grammar):
        self.MAX_LEN = MAX_LEN
        self.grammar = grammar
        self.reset()

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
        return mask

    def process_one_action(self, i, a):
        if a is not None:
            # 1. Apply the expansion from last prod rule
            this_rule = self.grammar.GCFG.productions()[a]
            if this_rule.lhs() != Nonterminal('Nothing'):
                # find the token to apply the expansion to, check it has the right lhs
                for this_index, this_token in enumerate(self.S[i]):
                    if not isinstance(this_token['token'], str):
                        break
                #assert (this_token['token'] == this_rule.lhs())

                # get the expansion
                new_tokens = [{'token': x} for x in this_rule.rhs()]

                # if the expansion is a new ring, give it a ringID=t
                if 'ring' in this_token['token']._symbol:
                    ringID = self.t
                    ring_size = 1
                # if the token to be replaced already has a ringID, propagate it
                elif 'ringID' in this_token:
                    ringID = this_token['ringID']
                    ring_size = this_token['ring_size'] + 1
                else:
                    ringID = None

                if ringID is not None:
                    for x in new_tokens:
                        if isinstance(x['token'], str):
                            continue
                        # ringID doesn't propagate to branches
                        if x['token']._symbol == 'branch':
                            continue
                        # nor beyond the end of the ring
                        if 'final' in this_token['token']._symbol \
                                and 'bond' in x['token']._symbol:
                            continue

                        x['ringID'] = ringID
                        x['ring_size'] = ring_size

                # do the replacement
                self.S[i] = self.S[i][:this_index] + new_tokens + self.S[i][this_index + 1:]

            # 2. generate masks for next prod rule
            # find the index of the next token to expand, which is the first nonterminal in sequence
        for this_index, this_token in enumerate(self.S[i]):
            if not isinstance(this_token['token'], str):
                break
            this_token = {'token': nltk.grammar.Nonterminal('Nothing')}

        if this_token['token'] == nltk.grammar.Nonterminal('Nothing'):
            # we only get to this point if the sequence is fully expanded
            return self.grammar.masks[self.grammar.lhs_to_index[this_token['token']], :]

            # get the formal grammar mask
        self.grammar_mask = self.grammar.masks[self.grammar.lhs_to_index[this_token['token']], :]

        # get the terminal distance mask
        term_distance = sum([self.grammar.terminal_dist(x['token']) for x in self.S[i]
                             if not isinstance(x['token'], str)])
        steps_left = self.MAX_LEN - self.t - 1
        self.terminal_mask = np.zeros_like(self.grammar_mask)
        self.terminal_mask[self.grammar.rule_d_term_dist < steps_left - term_distance - 1] = 1

        if 'ringID' in this_token:
            # if we're expanding a ring numeric token:
            if 'num' in this_token['token']._symbol:
                full_num_ID = (i, this_token['ringID'], this_token['token']._symbol)
                if full_num_ID in self.ring_num_map:
                    # this is the closing numeral of a ring
                    num_to_use = self.ring_num_map.pop(full_num_ID)
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
                    num_to_use = nt
                    self.ring_num_map[full_num_ID] = nt

                self.ring_mask = np.array(
                    [1 if a.rhs()[0] == num_to_use else 0 for a in self.grammar.GCFG.productions()])
            else:
                self.ring_mask = np.ones_like(self.grammar_mask)
                if this_token['ring_size'] > 4:
                    # block out cycle continuation
                    for k in self.grammar.cycle_continues:
                        self.ring_mask[k] = 0
                if this_token['ring_size'] < 4:
                    # block out cycle finish
                    for k in self.grammar.cycle_ends:
                        self.ring_mask[k] = 0

        else:
            self.ring_mask = np.ones_like(self.grammar_mask)

        return self.grammar_mask * self.terminal_mask * self.ring_mask
