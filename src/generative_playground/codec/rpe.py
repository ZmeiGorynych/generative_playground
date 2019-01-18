from nltk.tree import *
import nltk
from generative_playground.codec.cyk import *

from generative_playground.codec.grammar_codec import GrammarModel
from generative_playground.molecules.model_settings import get_settings

settings = get_settings(True, 'new')
from generative_playground.codec.smiles_grammar_new_2 import grammar_string_zinc_new
from generative_playground.codec.grammar_helper import GrammarHelper
grammar_string = grammar_string_zinc_new


fname = '../data/250k_rndm_zinc_drugs_clean.smi'

with open(fname) as f:
    smiles = f.readlines()

for i in range(len(smiles)):
    smiles[i] = smiles[i].strip()

def parse(x, codec):
    # print(x)
    try:
        return next(codec._parser.parse(x))
    except Exception as e:
        #         print('fail!')
        #         print(e)
        return None

for _ in range(10):
    codec = GrammarModel(grammar=GrammarHelper(grammar_string)
    , tokenizer=settings['tokenizer'])



    tokens = [codec._tokenize(s) for s in smiles[:100]]

    parse_trees = [(t,parse(t, codec)) for t in tokens]
    tokens = [p[0] for p in parse_trees if p[1] is not None]
    parse_trees = [p[1] for p in parse_trees if p[1] is not None]


    GCFG = nltk.CFG.fromstring(grammar_string)
    prods = GCFG.productions()
    # assume the lhs of first rule is root token
    no_sing = eliminate_singular_rules(prods, prods[0].lhs())
    parseable = cyk_parser(prods, tokens)
    break
    print("average productions:", np.array([len(p.productions()) for p in parse_trees]).mean())

    rpe_dict ={}

    def get_rpe(tree, rpe_dict):
        these_tuples = [(tree.productions()[0], loc, child.productions()[0]) for loc,child in enumerate(tree) if isinstance(child,Tree)]
        for t in these_tuples:
            if t in rpe_dict:
                rpe_dict[t] += 1
            else: rpe_dict[t] = 1
        for subtree in tree:
            if isinstance(subtree, Tree):
                rpe_dict = get_rpe(subtree, rpe_dict)
        return rpe_dict
    for tree in parse_trees:
        rpe_dict = get_rpe(tree,rpe_dict)
    #print(rpe_dict)
    rpe_count = [(key, value) for key, value in rpe_dict.items()]
    rpe_count = sorted(rpe_count, key = lambda x:x[1],reverse = True)
    print(len(parse_trees))
    for x in rpe_count[:1]:
        print(x)

    first_rule, loc, second_rule = rpe_count[0][0]
    new_rhs = []
    count = 0
    for t in first_rule.rhs():
        if is_nonterminal(t) and count == loc:
            new_rhs += second_rule.rhs()
        else:
            new_rhs.append(t)

    print(first_rule)
    print(second_rule)
    print(new_rhs)
    new_prod = str(Production(first_rule.lhs(), new_rhs))
    print(new_prod)
    grammar_string = grammar_string.split('\n')
    new_prods = grammar_string[:1] + [new_prod] + grammar_string[1:]
    grammar_string ="\n".join([str(p).replace('\\\\','\\')  for p in new_prods])

print('done!')