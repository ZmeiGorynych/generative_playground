from generative_playground.codec.hyperedge import HyperGraphFragment, to_mol
from generative_playground.molecules.model_settings import get_settings
from rdkit.Chem import MolFromSmiles, AddHs, MolToSmiles, RemoveHs
import networkx as nx
import copy


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


def edge_match(e1, e2):
    return e1['bond_type'] == e2['bond_type']


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

def get_neighbor_atoms(mol, my_index):
    neighbor_atoms = [(my_index, n.GetIdx()) for n in mol.GetAtomWithIdx(my_index).GetNeighbors()]
    return neighbor_atoms


def build_junction_tree(mol):
    atom_rings = mol.GetRingInfo().AtomRings()
    bond_rings = mol.GetRingInfo().BondRings()
    atoms_left = set(atom.GetIdx() for atom in mol.GetAtoms())

    def junction_tree_stage(my_start_atoms, parent_bonds=[]):
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

        # determine my children:
        neighbor_atoms = []
        for atom_idx in my_atoms:
            pre_neighbors = get_neighbor_atoms(mol, atom_idx)
            neighbor_atoms += [n for n in pre_neighbors if n[1] not in my_atoms]

        # maps the internal index this node uses for the bond to rdkit's index of this bond
        neighbor_bonds = {i: (n, mol.GetBondBetweenAtoms(n[0],n[1]).GetIdx())
                       for i, n in enumerate(neighbor_atoms)}

        child_bonds = {key:value for key, value in neighbor_bonds.items() if value[1] not in parent_bonds}

        me = {'atoms': my_atoms, 'child_bonds': child_bonds}

        child_bond_groups = []
        processed = []
        # now go over the neighbor atoms checking if they're part of another ring
        for i, child_bond in child_bonds.items():

            if i in processed: # already  processed
                continue

            processed.append(i)
            this_group = [i]
            for bond_ring in bond_rings:
                if child_bond[1] in bond_ring:
                    # what other bonds are part of the same ring?
                    for j, bond_idx in child_bonds.items():
                        if bond_idx[1] in bond_ring and j not in this_group:
                            this_group.append(j)
                            processed.append(j)


            child_bond_groups.append(this_group)

        # and now the recursive call
        children = {}
        for group in child_bond_groups:
            next_parent_bonds = [child_bonds[g][1] for g in group]
            next_start_atoms = [child_bonds[g][0][1] for g in group]
            children[tuple(group)] = junction_tree_stage(next_start_atoms, next_parent_bonds)

        me['children'] = children
        return me

    # could be smarter about the starting atom, but let's keep it simple for now
    return junction_tree_stage([0])


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
    # test = components[0]

    mol = MolFromSmiles(L[0])
    tree = build_junction_tree(mol)
    # look at isomorphisms between them
    print('aaa')

    '''
    So how do we turn the molecule into a tree?
    1. Start at any vertex, let's suppose for simplcity it's not a ring.
    2. The children are all connected components, so top node is atom + list
    of edges wth type
    3. If nearest atom is not part of a ring, just take its children - nice and iterative
    4. If child is a ring: fine, that ring is the child. Its children are the branches and any further rings attached to it


    '''