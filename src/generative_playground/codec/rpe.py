import copy
from collections import defaultdict
from nltk.tree import *
from generative_playground.codec.cyk import *
from rdkit.Chem import MolFromSmiles
from .hypergraph_grammar import HypergraphTree, apply_rule
from .hypergraph_parser import hypergraph_parser
from .hypergraph import replace_nonterminal



def parse(x, codec):
    try:
        return next(codec._parser.parse(x))
    except Exception as e:
        return None

def get_parse_trees(codec, smiles):
    tokens = [codec._tokenize(s) for s in smiles]
    parse_trees = [(t,parse(t, codec)) for t in tokens]
    # tokens = [p[0] for p in parse_trees if p[1] is not None]
    parse_trees = [p[1] for p in parse_trees if p[1] is not None]
    return parse_trees

# find all rule combinations
def get_rpe(tree, rpe_dict):
    these_tuples = [(tree.productions()[0], loc, child.productions()[0]) for loc,child in enumerate(tree) if isinstance(child,Tree)]
    for t in these_tuples:
        if t in rpe_dict:
            rpe_dict[t] += 1
        else:
            rpe_dict[t] = 1
    for subtree in tree:
        if isinstance(subtree, Tree):
            rpe_dict = get_rpe(subtree, rpe_dict)
    return rpe_dict


def apply_substitution_rule(tree, rule_pair):
    if not isinstance(tree, Tree):
        return tree
    first_rule, loc, second_rule = rule_pair
    if tree.productions()[0] == first_rule and tree[loc].productions()[0] == second_rule:
        new_children = list(tree[:loc]) + list(tree[loc]) + list(tree[(loc+1):])
    else:
        new_children = [child for child in tree]

    transformed_children = [apply_substitution_rule(child, rule_pair) for child in new_children]
    out = Tree(tree._label, transformed_children)
    out.__str__ = lambda tr: '(' + tr._label + '(' + str(list(tr)) + ')'
    return out

def apply_substitution_rules(tree, rules):
    for rule_pair in rules:
        tree = apply_substitution_rule(tree, rule_pair)
    return tree

def collapse_rule_pair(rule_pair):
    first_rule, loc, second_rule = rule_pair
    new_rhs = first_rule.rhs()[:loc] + second_rule.rhs() + first_rule.rhs()[loc+1:]
    new_rule = Production(first_rule.lhs(), new_rhs)
    return new_rule

def extract_popular_pairs(parse_trees, num_rules):
    lens = np.array([len(p.productions()) for p in parse_trees])
    print("initial average productions:", lens.mean(), 'max:', lens.max())

    new_rules = []
    for i in range(num_rules):
        print('iteration', i)
        rpe_dict = {}

        # collect statistics over all trees
        for tree in parse_trees:
            rpe_dict = get_rpe(tree,rpe_dict)

        # find the most popular rule pair
        rpe_count = [(key, value) for key, value in rpe_dict.items()]
        rpe_count = sorted(rpe_count, key = lambda x:x[1],reverse = True)
        best_rule_pair = rpe_count[0][0]
        for x in rpe_count[:1]:
            print(x)

        # create a new rule by collapsing these two rules

        new_rules.append(best_rule_pair)
        print(collapse_rule_pair(best_rule_pair))

        parse_trees = [apply_substitution_rule(tree, best_rule_pair) for tree in parse_trees]

        lens = np.array([len(tree.productions()) for tree in parse_trees])
        print('average num productions', np.mean(lens), 'max', np.max(lens))

    return new_rules


class RPEParser:
    def __init__(self, parser, rule_pairs):
        '''
        Adds rpe post-processing to a parse tree
        :param parser: accepts an iterable of token lists, returns a generator of parse trees
        :param rule_pairs: rpe rule pairs to collapse
        :return: another iterable
        '''
        self._parser = parser
        self._rule_pairs = rule_pairs
    # tokens = codec._tokenize(x)
    def parse(self, tokens):
        '''
        :param t: A list of tokens
        :return: a generator which returns the parsed value (to behave like nltk.ChartParser)
        '''
        try:
            tree = next(self._parser.parse(tokens))
        except:
            tree = None

        if tree is not None:
            for rule_pair in self._rule_pairs:
                tree = apply_substitution_rule(tree, rule_pair)

        def out_gen():
            yield tree

        return out_gen()


class HypergraphRPEParser:
    def __init__(self, grammar, rule_pairs):
        '''
        Adds rpe post-processing to a hypergraph parse tree
        :param grammar: HypergraphGrammar containing the current rule_pairs
        :param rule_pairs: rpe rule pairs to collapse
        :return: another iterable
        '''
        self.grammar = grammar
        self.rule_pairs = rule_pairs

    def parse(self, x):
        '''
        :param smiles: A valid smiles string
        :return: a hypergraph tree after applying RPE
        '''
        if isinstance(x, str):
            molecule = MolFromSmiles(x)
            tree = hypergraph_parser(molecule)
            norm_tree = self.grammar.normalize_tree(tree)
        else:
            norm_tree = x

        for rule_pair in self.rule_pairs:
            tree = apply_hypergraph_substitution(
                self.grammar, norm_tree, rule_pair
            )
        return tree

if __name__ == '__main__':
    from generative_playground.molecules.model_settings import get_settings
    settings = get_settings(True, 'new')# True)  #

    with open(settings['source_data']) as f:
        smiles = f.readlines()

    for i in range(len(smiles)):
        smiles[i] = smiles[i].strip()

    codec = settings['codec']
    trees = get_parse_trees(codec, smiles[:100])
    rule_pairs = extract_popular_pairs(trees, 10)

    # and a test parse
    rpe_parser = RPEParser(codec._parser, rule_pairs)
    for smile in smiles:
        tokens = codec._tokenize(smile)
        new_tree = next(rpe_parser.parse(tokens))
        if new_tree is not None:
            break

    print(new_tree, len(new_tree.productions()), len(trees[0].productions()))


# new_rhs = []
# count = 0
# for t in first_rule.rhs():
#     if is_nonterminal(t) and count == loc:
#         new_rhs += second_rule.rhs()
#     else:
#         new_rhs.append(t)
#
# print(first_rule)
# print(second_rule)
# print(new_rhs)
# new_prod = str(Production(first_rule.lhs(), new_rhs))
# print(new_prod)
# grammar_string = grammar_string.split('\n')
# new_prods = grammar_string[:1] + [new_prod] + grammar_string[1:]
# grammar_string ="\n".join([str(p).replace('\\\\','\\')  for p in new_prods])

print('done!')
