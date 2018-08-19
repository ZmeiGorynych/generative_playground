import nltk
import numpy as np
import six
import pdb
from nltk.grammar import Nonterminal

from generative_playground.codec.smiles_grammar_new_2 import grammar_string_zinc_new

# the zinc grammar
grammar_string_zinc = """smiles -> chain
atom -> bracket_atom
atom -> aliphatic_organic
atom -> aromatic_organic
aliphatic_organic -> 'B'
aliphatic_organic -> 'C'
aliphatic_organic -> 'N'
aliphatic_organic -> 'O'
aliphatic_organic -> 'S'
aliphatic_organic -> 'P'
aliphatic_organic -> 'F'
aliphatic_organic -> 'I'
aliphatic_organic -> 'Cl'
aliphatic_organic -> 'Br'
aromatic_organic -> 'c'
aromatic_organic -> 'n'
aromatic_organic -> 'o'
aromatic_organic -> 's'
bracket_atom -> '[' BAI ']'
BAI -> isotope symbol BAC
BAI -> symbol BAC
BAI -> isotope symbol
BAI -> symbol
BAC -> chiral BAH
BAC -> BAH
BAC -> chiral
BAH -> hcount BACH
BAH -> BACH
BAH -> hcount
BACH -> charge class
BACH -> charge
BACH -> class
symbol -> aliphatic_organic
symbol -> aromatic_organic
isotope -> DIGIT
isotope -> DIGIT DIGIT
isotope -> DIGIT DIGIT DIGIT
DIGIT -> '1'
DIGIT -> '2'
DIGIT -> '3'
DIGIT -> '4'
DIGIT -> '5'
DIGIT -> '6'
DIGIT -> '7'
DIGIT -> '8'
chiral -> '@'
chiral -> '@@'
hcount -> 'H'
hcount -> 'H' DIGIT
charge -> '-'
charge -> '-' DIGIT
charge -> '-' DIGIT DIGIT
charge -> '+'
charge -> '+' DIGIT
charge -> '+' DIGIT DIGIT
bond -> '-'
bond -> '='
bond -> '#'
bond -> '/'
bond -> '\\'
ringbond -> DIGIT
ringbond -> bond DIGIT
branched_atom -> atom
branched_atom -> atom RB
branched_atom -> atom BB
branched_atom -> atom RB BB
RB -> RB ringbond
RB -> ringbond
BB -> BB branch
BB -> branch
branch -> '(' chain ')'
branch -> '(' bond chain ')'
chain -> branched_atom
chain -> chain branched_atom
chain -> chain bond branched_atom
Nothing -> None"""

grammar_string_eq = """S -> S '+' T
S -> S '*' T
S -> S '/' T
S -> T
T -> '(' S ')'
T -> 'sin(' S ')'
T -> 'exp(' S ')'
T -> 'x'
T -> '1'
T -> '2'
T -> '3'
Nothing -> None"""

class GrammarHelper:
    def __init__(self, grammar_str, molecule_tweak=False):
        self.GCFG = nltk.CFG.fromstring(grammar_str)
        self.start_index = self.GCFG.productions()[0].lhs()

        # collect all lhs symbols, and the unique set of them
        all_lhs = [a.lhs() for a in self.GCFG.productions()]
        self.lhs_list = []

        self.cycle_ends = []
        self.cycle_continues = []
        for ip, p in enumerate(self.GCFG.productions()):
            if p.lhs() in [Nonterminal('cycle_bond'), Nonterminal('cycle_double_bond')]:
                if any(['num1' in pp._symbol for pp in p.rhs()]): #Nonterminal('num1') in p.rhs()
                    self.cycle_ends.append(ip)
                else:
                    self.cycle_continues.append(ip)

        for a in all_lhs:
            if a not in self.lhs_list:
                self.lhs_list.append(a)
        self.lhs_to_index = {lhs: ix for ix, lhs in enumerate(self.lhs_list)}
        self.D = len(self.GCFG.productions())
        self.term_dist = {}

        # TODO: pull the numeric tokens from the grammar itself
        self.numeric_tokens = [str(i) for i in range(1,10)] + ['%' + str(i) for i in range(10,50)]

        # this map tells us the rhs symbol indices for each production rule
        self.rhs_map = [None]*self.D
        count = 0
        for count,a in enumerate(self.GCFG.productions()):
            self.rhs_map[count] = []
            for b in a.rhs():
                if not isinstance(b,six.string_types):
                    s = b#.symbol()
                    self.rhs_map[count].extend(list(np.where(np.array(self.lhs_list) == s)[0]))


        self.masks = np.zeros((len(self.lhs_list),self.D))

        #count = 0
        # this tells us for each lhs symbol which productions rules should be masked
        for count,sym in enumerate(self.lhs_list):
            is_in = np.array([a == sym for a in all_lhs], dtype=int).reshape(1,-1)
            self.masks[count] = is_in
            #count = count + 1

        # go over all production rules, collect all nonterminal symbols
        for p in self.GCFG.productions():
            self.term_dist[p.lhs()] = float('inf')

        # iteratively determine minimum distance to terminal symbol
        while any([val == float('inf') for key,val in self.term_dist.items()]):
            for p in self.GCFG.productions():
                this_dist = sum([self.terminal_dist(x) for x in p.rhs()]) + 1
                self.term_dist[p.lhs()] = min(self.term_dist[p.lhs()], this_dist)

        # determine all transition rules that decrease the total terminal distance
        self.terminal_mask = np.zeros((1,self.D))
        self.rule_d_term_dist = np.zeros(self.D)
        self.max_term_dist_increase = 0
        for ip, p in enumerate(self.GCFG.productions()):
            old_dist = self.terminal_dist(p.lhs())
            new_dist = sum([self.terminal_dist(s) for s in p.rhs()])
            self.rule_d_term_dist[ip] = new_dist - old_dist
            if old_dist > new_dist:
                self.terminal_mask[0,ip] = 1
            if nltk.Nonterminal('class') not in p.rhs():
                self.max_term_dist_increase = max(self.max_term_dist_increase, new_dist-old_dist)

        # this tells us the indices where the masks are equal to 1
        index_array = []
        for i in range(self.masks.shape[1]):
            index_array.append(np.where(self.masks[:,i]==1)[0][0])
        self.ind_of_ind = np.array(index_array)

        max_rhs = max([len(l) for l in self.rhs_map])

        self.ind_to_lhs_ind = -np.ones(len(all_lhs), dtype=int)
        for i, a in enumerate(all_lhs):
            for ind, un_a in enumerate(self.lhs_list):
                if a == un_a:
                    self.ind_to_lhs_ind[i] = ind

        if molecule_tweak:
            # rules 29 and 31 aren't used in the zinc data so we
            # 0 their masks so they can never be selected
            self.masks[:,29] = 0
            self.masks[:,31] = 0

    def terminal_dist(self,x):
        if type(x)==nltk.grammar.Nonterminal:
            if x._symbol == 'class':
                #a nonterminal symbol that never gets used - why was it even included
                return float('inf')
            elif x._symbol == 'None':
                return 0
            else:
                return self.term_dist[x]
        else:
            return 0


grammar_zinc = GrammarHelper(grammar_string_zinc, molecule_tweak=True)
grammar_zinc_new = GrammarHelper(grammar_string_zinc_new)
grammar_eq = GrammarHelper(grammar_string_eq)
