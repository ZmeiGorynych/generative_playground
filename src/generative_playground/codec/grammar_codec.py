import re
import nltk
import numpy as np

import generative_playground.codec.grammar_helper as grammar_helper
from generative_playground.codec.parent_codec import GenericCodec
from generative_playground.codec.rpe import RPEParser


class CFGrammarCodec(GenericCodec):
    def __init__(self,
                 model=None,
                 max_len=None,
                 grammar=None,
                 tokenizer=None,
                 rpe_rules=None):
        """ Load the (trained) zinc encoder/decoder, grammar model. """
        super().__init__()
        # self.set_model(model)
        self.grammar = grammar
        # self._model = model
        self._tokenize = tokenizer
        self.MAX_LEN = max_len
        self._productions = self.grammar.GCFG.productions()
        self._prod_map = {}
        for ix, prod in enumerate(self._productions):
            self._prod_map[prod] = ix
        if rpe_rules is None:
            self._parser = nltk.ChartParser(self.grammar.GCFG)
        else:
            self._parser = RPEParser(nltk.ChartParser(self.grammar.GCFG), rpe_rules)
        self._n_chars = len(self._productions)

    def feature_len(self):
        return len(self._productions)

    def strings_to_actions(self, smiles):
        """ Encode a list of smiles strings into the latent space """
        assert type(smiles) == list or type(smiles) == tuple
        tokens = [self._tokenize(s.replace('-c', 'c')) for s in smiles]
        parse_trees = [next(self._parser.parse(t)) for t in tokens]
        productions_seq = [tree.productions() for tree in parse_trees]
        actions = [[self._prod_map[prod] for prod in entry] for entry in productions_seq]
        # now extend them to max length
        actions = np.array([a + [self._n_chars - 1] * (self.MAX_LEN - len(a)) for a in actions])
        return actions

    def decode_from_actions(self, actions):
        '''
        Takes a batch of action sequences, applies grammar
        :param actions: batch_size x seq_length LongTensor or ndarray(ints)
        :return: list of strings
        '''
        # Convert from one-hot to sequence of production rules
        prod_seq = [[self._productions[actions[index, t]]
                     for t in range(actions.shape[1])]
                    for index in range(actions.shape[0])]
        out = []
        for ip, prods in enumerate(prod_seq):
            grammar_out = prods_to_eq(prods)
            # as NLTK doesn't allow empty strings as terminals, we used 'h' for implicit hydrogen
            # now need to purge these
            grammar_out = grammar_out.replace('h', '')
            out.append(grammar_out)

        return out


def eq_tokenizer(s):
    funcs = ['sin', 'exp']
    for fn in funcs: s = s.replace(fn + '(', fn + ' ')
    s = re.sub(r'([^a-z ])', r' \1 ', s)
    for fn in funcs: s = s.replace(fn, fn + '(')
    return s.split()


def get_zinc_tokenizer(cfg):
    long_tokens = [a for a in cfg._lexical_index.keys() if
                   len(a) > 1]  # filter(lambda a: len(a) > 1, cfg._lexical_index.keys())
    replacements = ['$', '%', '^', '&', '!', '_', '|']
    # TODO: revisit this when parsing into the new grammar
    # assert len(long_tokens) <= len(replacements)
    for token in replacements:
        # assert not cfg._lexical_index.has_key(token)
        assert not token in cfg._lexical_index

    def tokenize(smiles):
        for i, token in enumerate(long_tokens):
            if token in smiles:
                smiles = smiles.replace(token, replacements[i])
        tokens = []
        for token in smiles:
            try:
                ix = replacements.index(token)
                tokens.append(long_tokens[ix])
            except:
                tokens.append(token)
        return tokens

    return tokenize


zinc_tokenizer = get_zinc_tokenizer(grammar_helper.grammar_zinc.GCFG)
zinc_tokenizer_new = get_zinc_tokenizer(grammar_helper.grammar_zinc_new.GCFG)


def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == nltk.grammar.Nonterminal('Nothing'):
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix + 1:]
                break
    try:
        return ''.join(seq)
    except:
        return 'ran_out_of_max_length'
        # raise Exception("We've run out of max_length but still have nonterminals: something is wrong here...")
