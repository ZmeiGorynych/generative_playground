import nltk

class GrammarMaskGenerator:
    '''
    This is a discrete operation, so no need to inherit from nn.Module etc
    '''
    def __init__(self, MAX_LEN, grammar=None):
        self.S = None
        self.t = 0
        self.MAX_LEN = MAX_LEN
        self.grammar = grammar
        self._lhs_map ={lhs: ix for ix, lhs in enumerate(self.grammar.lhs_list)}

    def reset(self):
        self.S = None
        self.t = 0

    def __call__(self, last_action):
        '''
        Consumes one action at a time, responds with the mask for next action
        : param last_action: previous action, array of ints of len = batch_size; None for the very first step
        '''
        if self.t >= self.MAX_LEN:
            raise StopIteration("maximum sequence length exceeded for decoder")

        if last_action[0] is None:
            # first call
            self.S = [[self.grammar.start_index] for _ in range(len(last_action))]
            self.tdist_reduction = [False for _ in range(len(self.S))]
        else:
            # insert the non-terminals from last action into the stack in reverse order
            rhs = [[x for x in self.grammar.GCFG.productions()[sampled_ind].rhs()
                    if (type(x) == nltk.grammar.Nonterminal) and (str(x) != 'None')]
                        for sampled_ind in last_action]

            for ix, this_rhs in enumerate(rhs):
                self.S[ix] += [x for x in this_rhs[::-1]]

        # Have to calculate total terminal distance BEFORE we pop the next nonterminal!
        self.term_distance = [sum([self.grammar.terminal_dist(sym) for sym in s]) for s in self.S]

        # get the next nonterminal and look up the mask for it
        next_nonterminal = [self._lhs_map[pop_or_nothing(a)] for a in self.S]
        mask = self.grammar.masks[next_nonterminal]

        # add masking to make sure the sequence always completes
        # TODO: vectorize this
        for ix, s in enumerate(self.S):
            #term_distance = sum([self.grammar.terminal_dist(sym) for sym in s])
            if self.term_distance[ix] >= self.MAX_LEN - self.t - self.grammar.max_term_dist_increase - 1:
                self.tdist_reduction[ix] = True  # go into terminal distance reduction mode for that molecule
            if self.tdist_reduction[ix]:
                mask[ix] *= self.grammar.terminal_mask[0]


        self.t += 1
        return mask#.astype(int)


def pop_or_nothing(S):
    if len(S):
        return S.pop()
    else:
        return nltk.grammar.Nonterminal('Nothing')