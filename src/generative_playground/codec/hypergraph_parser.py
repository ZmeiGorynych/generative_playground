import copy
from collections import OrderedDict
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdmolops import Kekulize
from networkx.algorithms.clique import enumerate_all_cliques

from generative_playground.codec.hypergraph import (
    HyperGraph,
    Node,
    HypergraphTree,
    replace_nonterminal,
)


def get_neighbors(mol, my_index):
    # returns a dict from all the bonds ids to pairs of respective atom ids
    bonds = OrderedDict()
    atom_pairs = [
        (my_index, n.GetIdx()) for n in mol.GetAtomWithIdx(my_index).GetNeighbors()
    ]
    for pair in atom_pairs:
        bonds[bond_id_from_atom_ids(mol, pair)] = pair
    return bonds


def bond_id_from_atom_ids(mol, atoms):
    return mol.GetBondBetweenAtoms(atoms[0], atoms[1]).GetIdx()


def build_junction_tree(mol):
    Kekulize(mol, clearAromaticFlags=True)
    atom_rings = mol.GetRingInfo().AtomRings()
    bond_rings = mol.GetRingInfo().BondRings()
    atoms_left = set(atom.GetIdx() for atom in mol.GetAtoms())

    def junction_tree_stage(my_start_atoms, parent_bond_inds=[]):
        # TODO: assert that start atoms are in parent bonds, if any
        # am I part of any rings?
        my_atoms = copy.copy(my_start_atoms)
        for ring in atom_rings:
            for atom_idx in my_start_atoms:
                if atom_idx in ring:
                    for other_idx in ring:
                        # don't insert ring atoms that were already parsed
                        if other_idx not in my_atoms and other_idx in atoms_left:
                            my_atoms.append(other_idx)

        # this is a check that we never assign the same atom to two nodes
        for idx in my_atoms:
            assert idx in atoms_left
            atoms_left.discard(idx)

        # determine all my bonds
        my_bonds_pre = OrderedDict()
        for atom_idx in my_atoms:
            my_bonds_pre.update(get_neighbors(mol, atom_idx))

        # enumerate my bonds
        my_bonds = my_bonds_pre
        internal_bonds = OrderedDict(
            [
                (key, value)
                for key, value in my_bonds.items()
                if value[0] in my_atoms and value[1] in my_atoms
            ]
        )
        parent_bonds = OrderedDict(
            [(key, value) for key, value in my_bonds.items() if key in parent_bond_inds]
        )
        child_bonds = OrderedDict(
            [
                (key, value)
                for key, value in my_bonds.items()
                if key not in internal_bonds and key not in parent_bonds
            ]
        )

        me = {"atoms": my_atoms, "bonds": my_bonds, "parent_bonds": parent_bonds}

        child_bond_groups = []
        processed = []
        # now go over the neighbor atoms checking if they're part of another ring
        for child_bond, child_bond_atoms in child_bonds.items():

            if child_bond in processed:  # already  processed
                continue

            processed.append(child_bond)
            this_group = [child_bond]
            for bond_ring in bond_rings:
                if child_bond in bond_ring:
                    # what other bonds are part of the same ring?
                    for other_bond_idx in child_bonds.keys():
                        if (
                            other_bond_idx in bond_ring
                            and other_bond_idx not in this_group
                        ):
                            this_group.append(other_bond_idx)
                            processed.append(other_bond_idx)

            child_bond_groups.append(this_group)

        # and now the recursive call
        children = OrderedDict()
        for group in child_bond_groups:
            next_parent_bonds = group
            next_start_atoms = [child_bonds[g][1] for g in group]
            children[tuple(group)] = junction_tree_stage(
                next_start_atoms, next_parent_bonds
            )

        if len(children):
            me["children"] = children

        me["node"] = HyperGraph.from_tree_node(mol, me)

        return me

    # could be smarter about the starting atom, but let's keep it simple for now
    return junction_tree_stage([0])


def abstract_ring_atom(graph: HyperGraph, loc):
    graph.validate()
    atom = graph.node[loc]
    # replace ring atoms with nonterminals that just have the connections necessary for
    # the ring
    assert atom.is_terminal, "We only can abstract terminal atoms!"
    neighbors = [
        (edge_id, graph.other_end_of_edge(loc, edge_id)) for edge_id in atom.edge_ids
    ]
    # determine the nodes in the 'key structure', which is terminals, parent
    # nonterminal, and nonterminals connecting to more than one of the former
    internal_neighbor_ids = [
        x
        for x in neighbors
        if x[1] not in graph.child_ids() or len(graph.node[x[1]].edge_ids) > 1
    ]
    child_neighbors = [x for x in neighbors if x not in internal_neighbor_ids]
    if len(internal_neighbor_ids) >= 2 and len(internal_neighbor_ids) < len(neighbors):
        # create the edges between the parent placeholder and the atom being abstracted
        new_edges = [
            copy.deepcopy(graph.edges[edge_id[0]]) for edge_id in internal_neighbor_ids
        ]
        new_graph = HyperGraph()
        new_graph.add_edges(new_edges)
        parent = Node(edge_ids=[x for x in new_graph.edges.keys()], edges=new_edges)
        old_new_edge_map = {
            old[0]: new for old, new in zip(internal_neighbor_ids, parent.edge_ids)
        }
        child_edge_ids = [
            old_new_edge_map[edge_id] if edge_id in old_new_edge_map else edge_id
            for edge_id in atom.edge_ids
        ]

        # and move over all the edges from the abstracted atom to its children
        for edge_id in child_edge_ids:
            if edge_id not in old_new_edge_map.values():
                new_graph.edges[edge_id] = graph.edges[edge_id]
                del graph.edges[edge_id]
        new_graph.add_parent_node(parent)
        child = Node(
            data=atom.data,
            edge_ids=child_edge_ids,
            edges=[new_graph.edges[edge_id] for edge_id in child_edge_ids],
            is_terminal=True,
        )

        new_graph.add_node(child)

        # new hypergraph takes over all the children
        child_neighbor_nodes = [x[1] for x in child_neighbors]
        # determine which of the children we want to take over, we need the indices to
        # amend the tree later
        child_inds = []
        for ind, child_id in enumerate(graph.child_ids()):
            if child_id in child_neighbor_nodes:
                child_inds.append(ind)

        # and now purge these nodes from the original graph and append them to the
        # child, along with any edges
        for child_id in child_neighbor_nodes:
            new_graph.add_node(graph.node[child_id])
            del graph.node[child_id]
        new_graph.validate()

        # replace the atom with a nonterminal with just the connections for the ring
        replace_edge_ids = [edge_id[0] for edge_id in internal_neighbor_ids]
        replacement_node = Node(
            edge_ids=replace_edge_ids,
            edges=[graph.edges[edge_id] for edge_id in replace_edge_ids],
        )
        del graph.node[loc]
        graph.add_node(replacement_node)
        graph.validate()

        return new_graph, child_inds
    else:
        return None, None


def abstract_atom(graph: HyperGraph, loc):
    atom = graph.node[loc]
    # replace ring atoms with nonterminals that just have the connections necessary for
    # the ring
    assert atom.is_terminal, "We only can abstract terminal atoms!"
    new_edges = [copy.deepcopy(graph.edges[edge_id]) for edge_id in atom.edge_ids]

    new_graph = HyperGraph()
    new_graph.add_edges(new_edges)

    atom_copy = Node(
        edge_ids=[x for x in new_graph.edges.keys()],
        edges=[x for x in new_graph.edges.values()],
        is_terminal=True,
        data=atom.data,
    )
    parent = Node(edge_ids=atom_copy.edge_ids, edges=atom_copy.edges)
    new_graph.add_parent_node(parent)
    new_graph.add_node(atom_copy)
    new_graph.validate()

    # replace the original atom with a nonterminal matching the parent of the new graph
    replacement_node = Node(
        edge_ids=atom.edge_ids,
        edges=[graph.edges[edge_id] for edge_id in atom.edge_ids],
    )
    del graph.node[loc]
    graph.add_node(replacement_node)  # this makes sure it's appended at the end
    graph.validate()

    return new_graph


def abstract_ring_atoms(tree):
    new_children = [abstract_ring_atoms(x) for x in tree]

    if len(tree.node.terminal_ids()) >= 2:
        for t_id in tree.node.terminal_ids():
            # this modifies the tree.node object too, appending a nonterminal at the
            # end, instead of the abstracted atom
            new_child, subtree_inds = abstract_ring_atom(tree.node, t_id)
            if new_child is not None:
                grandchildren = []  # the abstracted node takes over some children
                for ind in sorted(subtree_inds, reverse=True):
                    grandchildren.append(new_children.pop(ind))
                child_tree = HypergraphTree(node=new_child, children=grandchildren)
                new_children.append(child_tree)
    new_tree = HypergraphTree(node=tree.node, children=new_children)
    return new_tree


def abstract_atoms(tree):
    check_tree_top_level(tree)
    new_tree = HypergraphTree(
        node=tree.node, children=[abstract_atoms(t) for t in tree]
    )
    check_tree_top_level(new_tree)
    for t_id in tree.node.terminal_ids():
        # this modifies the tree.node object too
        new_child = abstract_atom(tree.node, t_id)
        child_tree = HypergraphTree(node=new_child)
        new_tree.append(child_tree)
    check_tree_top_level(new_tree)
    return new_tree


def check_tree_top_level(new_tree):
    assert len(new_tree) == len(new_tree.node.child_ids())
    for child, child_id in zip(new_tree, new_tree.node.child_ids()):
        assert len(child.node.node[child.node.parent_node_id].edge_ids) == len(
            new_tree.node.node[child_id].edge_ids
        )


def dict_tree_to_graph_tree(tree):
    out_children = []
    if "children" in tree:
        for child in tree["children"].values():
            out_children.append(dict_tree_to_graph_tree(child))
    out_tree = HypergraphTree(node=tree["node"], children=out_children)
    return out_tree


def check_validity(child_node):
    for node_id, node in child_node.node.items():
        assert node_id == child_node.parent_node_id or len(node.data) > 0


def found_cliques(cliques):
    return not all([c == [] for c in cliques])


def filtered_cliques(graph, parent_node_id):
    return [c for c in enumerate_all_cliques(graph) if parent_node_id not in c]


def split_cliques(tree):
    hypergraph = tree.node
    children_nodes = hypergraph.child_ids()
    graph = hypergraph.to_nx()
    if hypergraph.parent_node_id is not None and children_nodes:
        cliques = filtered_cliques(graph, hypergraph.parent_node_id)
        # while found_cliques: TODO iterate until no more collapsing
        if found_cliques(cliques):
            clique = sorted(cliques, key=lambda x: len(x))[-1]
            clique_nodes = []
            clique_children = []
            clique_idxs = []
            for i, node_id in enumerate(hypergraph.child_ids()):
                if node_id in clique:
                    clique_nodes.append(hypergraph.node[node_id])
                    clique_children.append(tree[i])
                    clique_idxs.append(i)

            # Clean parent
            for node_id in clique:
                del hypergraph.node[node_id]
            for idx in sorted(clique_idxs, reverse=True):
                tree.pop(idx)

            # Add new non-terminal
            # Find external edges from clique
            clique_edges = set()
            for node in clique_nodes:
                clique_edges.update(set(node.edge_ids))

            # TODO Getting close but still not quite right with the edges
            # Need to see abstract_ring_atom above for inspiration.
            # In the new child, I need to add the edges in the correct place with new
            # ids.
            # For the parent, I need to point the one/two cut edges to the new
            # non-terminal
            external_edge_ids = []
            external_edges = []
            for node in hypergraph.node.values():
                for edge_id, edge in zip(node.edge_ids, node.edges):
                    if edge_id in clique_edges:
                        external_edge_ids.append(edge_id)
                        external_edges.append(edge)

            new_nt = Node(edge_ids=external_edges, edges=external_edges)
            hypergraph.add_node(new_nt)

            # Make new child from clique
            new_nt_child = copy.deepcopy(new_nt)
            new_child = HyperGraph()
            new_child.add_parent_node(new_nt_child)
            for node in clique_nodes:
                new_child.add_node(node)
                for edge in node.edges:
                    new_child.add_edges(node.edges)
            tree.append(HypergraphTree(new_child, clique_children))

            new_child.validate()
            tree.validate()

    for child in tree:
        split_cliques(child)
    return tree


def hypergraph_parser(mol):
    if type(mol) == str:
        mol = MolFromSmiles(mol)
    tree = build_junction_tree(mol)
    graph_tree = dict_tree_to_graph_tree(copy.deepcopy(tree))
    graph_tree_2a = abstract_ring_atoms(copy.deepcopy(graph_tree))
    graph_tree_2 = abstract_atoms(graph_tree_2a)
    # graph_tree_3 = split_cliques(graph_tree_2)
    return graph_tree_2


def graph_from_graph_tree(tree):
    """
    This constructs the graph directly from graph tree, bypassing the conversion to a
    linear rules list
    """
    assert isinstance(tree, HypergraphTree)
    this_node = tree.node
    child_nodes = [graph_from_graph_tree(x) for x in tree]
    # as we're recursively reconstructing
    for child_node in child_nodes:
        check_validity(child_node)

    child_ids = this_node.child_ids()
    assert len(child_ids) == len(child_nodes)
    for id, new_node in zip(child_ids, child_nodes):
        this_node = replace_nonterminal(this_node, id, new_node)

    check_validity(this_node)
    this_node.validate()
    return this_node


def tree_with_rule_inds_to_list_of_tuples(tree):
    tuples_list = []
    if tree.node.parent_node_id is None:  # root of the tree
        tuples_list.append((None, None, tree.node.rule_id))
    for child, child_id in zip(tree, tree.node.child_indices()):
        tuples_list.append((tree.node.rule_id, child_id, child.node.rule_id))
        tuples_list += tree_with_rule_inds_to_list_of_tuples(child)
    return tuples_list
