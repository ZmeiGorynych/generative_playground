from generative_playground.codec.character_codec import CharacterCodec
from generative_playground.codec.grammar_codec import CFGrammarCodec, zinc_tokenizer, eq_tokenizer, zinc_tokenizer_new
from generative_playground.codec.grammar_helper import grammar_zinc, grammar_eq, grammar_zinc_new
from generative_playground.codec.grammar_mask_gen import GrammarMaskGenerator
from generative_playground.codec.hypergraph_grammar import HypergraphGrammar
from generative_playground.codec.hypergraph_mask_generator import HypergraphMaskGenerator
from generative_playground.codec.mask_gen_new_2 import GrammarMaskGeneratorNew


def get_codec(molecules, grammar, max_seq_length):
    if grammar is True:
        grammar = 'classic'
    # character-based models
    if grammar is False:
        if molecules:
            charlist = ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[',
                        '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/',
                        '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']
        else:
            charlist = ['x', '+', '(', ')', '1', '2', '3', '*', '/', 's', 'i', 'n', 'e', 'p', ' ']

        codec = CharacterCodec(max_len=max_seq_length, charlist=charlist)
    elif grammar == 'classic':
        if molecules:
            codec = CFGrammarCodec(max_len=max_seq_length,
                           grammar=grammar_zinc,
                           tokenizer=zinc_tokenizer)
        else:
            codec = CFGrammarCodec(max_len=max_seq_length,
                           grammar=grammar_eq,
                           tokenizer=eq_tokenizer)
        codec.mask_gen = GrammarMaskGenerator(max_seq_length, codec.grammar)
    elif grammar == 'new':
        codec = CFGrammarCodec(max_len=max_seq_length,
                               grammar=grammar_zinc_new,
                               tokenizer=zinc_tokenizer_new)
        codec.mask_gen = GrammarMaskGeneratorNew(max_seq_length, codec.grammar)
    elif 'hypergraph' in grammar:
        grammar_cache = grammar.split(':')[1]
        assert grammar_cache is not None, "Invalid cached hypergraph grammar file:" + str(grammar_cache)
        codec = HypergraphGrammar.load(grammar_cache)
        codec.MAX_LEN = max_seq_length
        codec.calc_terminal_distance() # just in case it wasn't initialized yet
        codec.mask_gen = HypergraphMaskGenerator(max_seq_length, codec)
    assert hasattr(codec, 'PAD_INDEX')
    return codec