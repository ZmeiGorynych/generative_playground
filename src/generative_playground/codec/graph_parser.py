from generative_playground.codec.hyperedge import HyperGraphFragment, to_mol, Edge
from generative_playground.molecules.model_settings import get_settings
from collections import OrderedDict
from rdkit.Chem import MolFromSmiles, AddHs, MolToSmiles, RemoveHs
import networkx as nx
import copy, uuid


def get_rings(mol):
    bond_rings = mol.GetRingInfo().BondRings()
    ring_bonds = set()
    for x in bond_rings:
        for y in x:
            ring_bonds.add(y)

    hg = HyperGraphFragment.from_mol(mol)
    # drop all non-cycle bonds
    # ring_atoms = ring_info.GetRingAtoms(mol)

    edges = [x for x in hg.edges.keys()]
    for edge in edges:
        if edge not in ring_bonds:
            hg.edges.pop(edge)

    # find all connected components of size > 1
    graph = hg.to_nx()

    conn = [x for x in nx.connected_components(graph) if len(x) > 1]
    components = [graph.subgraph(chunk) for chunk in conn]
    return components


# def edge_match(e1, e2):
#     return e1['bond_type'] == e2['bond_type']


def add_if_new(old, candidate):
    tests = []
    for x in old:
        tests.append(nx.is_isomorphic(x, candidate, edge_match=edge_match))
    # print(tests)
    if not len(tests) or not any(tests):
        old.append(candidate)
        #print(len(old))


def collect_ring_types(smiles_list):
    components = []
    for smile in smiles_list:
        mol = MolFromSmiles(smile)
        candidates = get_rings(mol)
        for c in candidates:
            add_if_new(components, c)
    return components

def get_neighbors(mol, my_index):
    # returns a dict from all the bonds ids to pairs of respective atom ids
    bonds = OrderedDict()
    atom_pairs = [(my_index, n.GetIdx()) for n in mol.GetAtomWithIdx(my_index).GetNeighbors()]
    for pair in atom_pairs:
        bonds[bond_id_from_atom_ids(mol, pair)] = pair
    return bonds

def bond_id_from_atom_ids(mol, atoms):
    return mol.GetBondBetweenAtoms(atoms[0],atoms[1]).GetIdx()

def build_junction_tree(mol):
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
        internal_bonds = OrderedDict([(key, value) for key, value in my_bonds.items()
                                      if value[0] in my_atoms and value[1] in my_atoms])
        parent_bonds = OrderedDict([(key, value) for key, value in my_bonds.items()
                                      if key in parent_bond_inds])
        child_bonds = OrderedDict([(key, value) for key, value in my_bonds.items()
                                      if key not in internal_bonds and key not in parent_bonds])

        me = {'atoms': my_atoms,
              'bonds': my_bonds,
              'parent_bonds': parent_bonds}

        child_bond_groups = []
        processed = []
        # now go over the neighbor atoms checking if they're part of another ring
        for child_bond, child_bond_atoms in child_bonds.items():

            if child_bond in processed: # already  processed
                continue

            processed.append(child_bond)
            this_group = [child_bond]
            for bond_ring in bond_rings:
                if child_bond in bond_ring:
                    # what other bonds are part of the same ring?
                    for other_bond_idx in child_bonds.keys():
                        if other_bond_idx in bond_ring and other_bond_idx not in this_group:
                            this_group.append(other_bond_idx)
                            processed.append(other_bond_idx)

            child_bond_groups.append(this_group)

        # and now the recursive call
        children = OrderedDict()
        for group in child_bond_groups:
            next_parent_bonds = group
            next_start_atoms = [child_bonds[g][1] for g in group]
            children[tuple(group)] = junction_tree_stage(next_start_atoms, next_parent_bonds)

        if len(children):
            me['children'] = children

        me['node'] = HyperGraphFragment.from_tree_node(mol, me)

        return me

    # could be smarter about the starting atom, but let's keep it simple for now
    return junction_tree_stage([0])

def replace_nonterminal(orig_node, loc, new_node):
    node_to_replace = orig_node.node[loc]
    new_root_node = new_node.node[new_node.parent_node_id]
    new_node = copy.deepcopy(new_node)
    assert len(node_to_replace.edges) == len(new_root_node.edges)
    for edge1_id, edge2_id in zip(node_to_replace.edges, new_root_node.edges):
        edge1 = orig_node.edges[edge1_id]
        edge2 = new_node.edges[edge2_id]
        assert edge1.type == edge2.type
        new_edge = Edge(type=edge1.type, data={**edge1.data, **edge2.data})
        orig_node.edges[edge1_id] = new_edge
        del new_node.edges[edge2_id]
        # in the new segment, replace the edges leading to the root node with edges from the parent
        for node in new_node.node.values():
            for i in range(len(node.edges)):
                if node.edges[i] == edge2_id:
                    node.edges[i] = edge1_id

    orig_node.edges.update(new_node.edges)
    del new_node.node[new_node.parent_node_id]
    del orig_node.node[loc]
    orig_node.node.update(new_node.node)
    orig_node.validate()
    return orig_node

def graph_from_tree(tree):
    if 'children' in tree:
        this_node = tree['node']
        child_nodes = [graph_from_tree(x) for x in tree['children'].values()]
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
    else:
        child_node = tree['node']
        check_validity(child_node)
        return child_node

def check_validity(child_node):
    for node_id, node in child_node.node.items():
        assert node_id == child_node.parent_node_id or len(node.data) > 0

def apply_rule(start_graph, rule, loc=None):
    if loc == None:
        loc = start_graph.child_ids()[-1]
    start_graph = replace_nonterminal(start_graph, loc, rule)
    return start_graph

def apply_rules(rules):
    start_graph = rules[0]
    for num, rule in enumerate(rules[1:]):
        start_graph = apply_rule(start_graph, rule)
    return start_graph

def tree_to_rules_list(tree):
    rules_list = [tree['node']]
    if 'children' in tree:
        for child in reversed(tree['children'].values()):
            rules_list += tree_to_rules_list(child)
    return rules_list

'''
So when are rules equivalent:
nodes_match:
nonterminals: 
    if one is a parent node, another must also be
terminals:
    node1.data == node2.data
    
edges_match: if types match
'''

def hypergraphs_are_equivalent(graph1, graph2):
    def nodes_match(node1, node2):
        if node1['node'].is_terminal != node2['node'].is_terminal:
            return False
        elif node1['node'].is_terminal:
            # for terminals (atoms), data must match
            return node1['node'].data == node2['node'].data
        else:
            # parent nodes must be aligned
            return graph1.is_parent_node(node1['node']) == graph2.is_parent_node(node2['node'])

    def edges_match(edge1, edge2):
        return edge1['data'].type == edge2['data'].type

    return nx.is_isomorphic(graph1.to_nx(), graph2.to_nx(), edge_match=edges_match, node_match=nodes_match)



if __name__ == '__main__':
    settings = get_settings(molecules=True, grammar='new')
    thresh = 100
    # Read in the strings
    f = open(settings['source_data'], 'r')
    L = []
    for line in f:
        line = line.strip()
        L.append(line)
        if len(L) > thresh:
            break
    f.close()

    # components = collect_ring_types(L[:thresh])
    distinct_rules = []
    all_rule_count = 0
    for num, smile in enumerate(L[:100]):
        smile = MolToSmiles(MolFromSmiles(smile))
        mol = MolFromSmiles(smile)
        tree = build_junction_tree(mol)
        rules_list = tree_to_rules_list(copy.deepcopy(tree))
        # for rule in rules_list:
        #     if not any([hypergraphs_are_equivalent(rule, x) for x in distinct_rules]):
        #         distinct_rules.append(rule)
        # all_rule_count += len(rules_list)
        # root_node = tree['node']
        graph2 = apply_rules(rules_list)
        graph = graph_from_tree(tree)
        mol2 = to_mol(graph)
        smiles2 = MolToSmiles(mol2)
        mol3 = to_mol(graph2)
        smiles3 = MolToSmiles(mol3)
        print(smile)
        print(smiles2)
        print(smiles3)
    # look at isomorphisms between them
    print('aaa')



