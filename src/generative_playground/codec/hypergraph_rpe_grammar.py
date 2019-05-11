import copy
from collections import defaultdict

from rdkit.Chem import MolFromSmiles

from .hypergraph_grammar import HypergraphGrammar, HypergraphTree, apply_rule
from .hypergraph_parser import hypergraph_parser


class HypergraphRPEGrammar(HypergraphGrammar):
    def __init__(self, cache_file='tmp.pickle', max_len=None):
        super().__init__(cache_file, max_len)
        self.rule_pairs = []

    def rpe_compress_tree(self, x):
        '''
        Apply current rule_pairs to compress tree
        :param x: A valid smiles string or a normalized hypergraph tree
        :return: a hypergraph tree after applying RPE
        '''
        if isinstance(x, str):
            molecule = MolFromSmiles(x)
            tree = hypergraph_parser(molecule)
            tree = self.grammar.normalize_tree(tree)
        else:
            tree = x

        for rule_pair in self.rule_pairs:
            tree = self.apply_hypergraph_substitution(tree, rule_pair)

        return tree

    def apply_hypergraph_substitution(self, tree, rule_pair):
        root_rule_id, child_rule_id, nt_loc = rule_pair
        if (
            nt_loc < len(tree.node.child_ids())
            and tree.node.rule_id == root_rule_id
            and tree[nt_loc].node.rule_id == child_rule_id
        ):
            assert self.rules[root_rule_id] == tree.node
            assert self.rules[child_rule_id] == tree[nt_loc].node

            child = copy.deepcopy(tree[nt_loc])
            parent_node = copy.deepcopy(tree.node)
            child_node = copy.deepcopy(child.node)
            new_node = apply_rule(
                parent_node,
                child_node,
                loc=parent_node.child_ids()[nt_loc]
            )

            rule_parent = copy.deepcopy(self.rules[root_rule_id])
            rule_child = copy.deepcopy(self.rules[child_rule_id])
            new_node_2 = apply_rule(
                rule_parent,
                rule_child,
                loc=rule_parent.child_ids()[nt_loc]
            )
            assert new_node == new_node_2

            children = tree[:nt_loc] + tree[(nt_loc+1):] + child[:]


            new_node.rule_id, _ = self.rule_to_index(new_node)
            if len(self.rules) - 1 == new_node.rule_id:
                print('Added rule! - {}, {}'.format(new_node.rule_id, str(new_node)))

        else:
            new_node = tree.node
            children = tree[:]

        transformed_children = [
            self.apply_hypergraph_substitution(child, rule_pair)
            for child in children
        ]
        return HypergraphTree(new_node, transformed_children)

    def extract_popular_hypergraph_pairs(self, hypergraph_trees, num_rules):
        new_rules = []
        for i in range(num_rules):
            print('iteration', i)
            rule_pair_frequencies = defaultdict(int)
            for tree in hypergraph_trees:
                self.count_rule_pair_frequencies(tree, rule_pair_frequencies)

            best_pair = max(
                rule_pair_frequencies, key=rule_pair_frequencies.get
            )
            new_rules.append(best_pair)

            hypergraph_trees = [
                self.apply_hypergraph_substitution(tree, best_pair)
                for tree in hypergraph_trees
            ]

        return new_rules

    def count_rule_pair_frequencies(self, tree, rule_pair_frequencies):
        for nt_loc, child in enumerate(tree):
            rule_pair_frequencies[
                (tree.node.rule_id, child.node.rule_id, nt_loc)
            ] += 1
            self.count_rule_pair_frequencies(child, rule_pair_frequencies)

    def extract_rpe_pairs(self, smiles, num_iterations) -> None:
        trees = self.raw_strings_to_trees(smiles)
        new_rule_pairs = self.extract_popular_hypergraph_pairs(
            trees, num_iterations
        )
        self.rule_pairs.extend(new_rule_pairs)
        self.reset_rule_frequencies()
        self._count_rpe_rule_frequencies(smiles)

    def raw_strings_to_trees(self, smiles):
        return [
            self.rpe_compress_tree(norm_tree)
            for norm_tree in self._smiles_to_tree_gen(smiles)
        ]

    def raw_strings_to_actions(self, smiles):
        '''
        Convert a list of valid SMILES string to actions, applying RPE
        :param smiles: a list of valid SMILES strings
        :return:
        '''
        actions = []
        for norm_tree in self._smiles_to_tree_gen(smiles):
            tree = self.rpe_compress_tree(norm_tree)
            these_actions = [rule.rule_id for rule in tree.rules()]
            actions.append(these_actions)
        return actions

    def _count_rpe_rule_frequencies(self, smiles):
        trees = self.raw_strings_to_trees(smiles)
        self.count_rule_frequencies(trees)
