import copy
import networkx as nx
import uuid
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AddHs, MolToSmiles, RemoveHs, Kekulize

from collections import OrderedDict


# each edge is a FrozenDict with a random ID
# each node is an OrderedDict with value 'edges' containing a list of edges, and optional other values from rdkit, and has a random ID too
class Node:
    def __init__(self, edges=[], is_terminal=False, data={}):
        self.edges = edges
        self.data = data
        self.is_terminal = is_terminal


class Edge:
    def __init__(self, type, data={}):
        self.type = type
        self.data = data


# so changes: use Bond class
# don't store endpoints in bonds, that just creates a mess! They're stored in the atoms!
class HyperGraphFragment:
    def __init__(self):
        self.node = OrderedDict()  # of Nodes
        self.edges = OrderedDict() # of edges
        self.parent_node_id = None  # The node we'll match/merge when expanding

    def nonterminal_ids(self):
        return [key for key, value in self.node.items() if value.is_terminal is False]

    def child_ids(self):
        return [key for key in self.nonterminal_ids() if key != self.parent_node_id]

    def is_parent_node(self, x):
        if self.parent_node_id is None:
            return False
        return self.node[self.parent_node_id] == x


    def validate(self):
        edge_count = {}
        edge_lists = [node.edges for node in self.node.values()]
        for node in edge_lists:
            for edge_id in node:
                if edge_id not in edge_count:
                    edge_count[edge_id] = 1
                else:
                    edge_count[edge_id] += 1

        for count in edge_count.values():
            assert count == 2

        assert set(edge_count.keys()) == set(self.edges.keys())

    def to_nx(self):
        # create a networkx Graph from self, ignoring any open edges
        # the main information that gets lost hereby is the ordering, so stereochemistry
        # self.check_validity()
        # assert len(self.open_edges) == 0
        G = nx.Graph()
        for node_id, node in self.node.items():
            G.add_node(node_id, node = node)

        for edge_id, edge in self.edges.items():
            first, second = self.node_ids_from_edge_id(edge_id)
            G.add_edge(first, second, id=edge_id, data=edge)
        return G

    def node_ids_from_edge_id(self, edge_id):
        my_nodes=[]
        for node_id, node in self.node.items():
            if edge_id in node.edges:
                my_nodes.append(node_id)
        return my_nodes

    # TODO: redo this so it also works when self has open edges, possibly with lhs_node - or does this work already?
    def replace_node(self, lhs_node, graph):
        assert lhs_node['edges'] == len(graph.open_edges)
        self.node = [node_ for node_ in self.node if node_ != lhs_node]
        edge_map = {old_edge: new_edge for old_edge, new_edge in zip(graph.open_edges, lhs_node['edges'])}
        for node_ in graph.nodes:
            new_node = copy.copy(node_)
            # any open edges are mapped to the edges the removed node had
            new_node['edges'] = [edge_map[edge] if edge in edge_map else edge for edge in node_['edges']]
            self.node.append(new_node)



    @classmethod
    def from_rule(Class, rule):
        pass


    @classmethod
    def from_mol(Class, mol):
        # make the Hs explicit, kekulize, so the graph is easier to work with
        Kekulize(mol, clearAromaticFlags=True)
        AddHs(mol)

        self = Class()
        # we'll use rdkit's indices as IDs for both nodes and edges
        for atom in mol.GetAtoms():
            assert atom.GetIdx() not in self.node
            this_node_edges = [bond.GetIdx() for bond in atom.GetBonds()]
            this_node = Node(edges=this_node_edges,
                             data=atom_to_properties(atom), is_terminal=True)
            self.node[atom.GetIdx()] = this_node

        for bond in mol.GetBonds():
            assert bond.GetIdx() not in self.edges
            this_bond = Edge(type=bond.GetBondType(),
                             data=bond_to_properties(bond))
            self.edges[bond.GetIdx()] = this_bond

        return self

    @classmethod
    def from_tree_node(Class, mol, tree_node):
        '''
        Constructs a graph from a junction tree node, with nonterminals for parent and children
        :param tree_node:
        :return:
        '''
        '''
        
        Add a 'parent' node, store its index
        Add child nodes from ['children'].keys()
        add all edges just as they are
        make sure to generate uids for nodes and edges as we create them
        '''
        self = Class()

        atom_id_map = OrderedDict()
        bond_id_map = OrderedDict()
        atoms_to_bonds = OrderedDict()
        for bond_old_id, bond_atoms in tree_node['bonds'].items():
            bond_id = uuid.uuid4()
            assert bond_id not in self.edges
            bond_id_map[bond_old_id] = bond_id
            bond = mol.GetBonds()[bond_old_id]
            for old_atom_id in bond_atoms:
                if old_atom_id not in atom_id_map:
                    atom_id_map[old_atom_id] = uuid.uuid4()
                if atom_id_map[old_atom_id] not in atoms_to_bonds:
                    atoms_to_bonds[atom_id_map[old_atom_id]] = []
                atoms_to_bonds[atom_id_map[old_atom_id]].append(bond_id)

            this_bond = Edge(type=bond.GetBondType(),
                             data=bond_to_properties(bond))
            self.edges[bond_id] = this_bond

        for old_atom_id, new_atom_id in atom_id_map.items():
            if old_atom_id in tree_node['atoms']:
                atom_properties = atom_to_properties(mol.GetAtoms()[old_atom_id])
                this_node = Node(edges=atoms_to_bonds[new_atom_id],
                                data=atom_properties,
                                is_terminal=True)
                self.node[new_atom_id] = this_node

        # now create the nonterminal nodes from bunches of loose bonds
        def create_node_from_bondlist(bond_list): # create a node from a bunch of loose bonds
            if len(bond_list):
                new_node_id = uuid.uuid4()
                my_edges = []
                for old_bond_id in bond_list:
                    my_edges.append(bond_id_map[old_bond_id])
                    for x in tree_node['bonds'][old_bond_id]: # remove the atom placeholder created when we created the edges
                        if x not in tree_node['atoms']:
                            del atoms_to_bonds[atom_id_map[x]]

                new_node = Node(edges=my_edges,
                                 data=OrderedDict(),
                                 is_terminal=False)
                self.node[new_node_id] = new_node
                return new_node_id
            else:
                return None

        self.parent_node_id = create_node_from_bondlist(tree_node['parent_bonds'].keys())
        if 'children' in tree_node:
            for child_group in tree_node['children'].keys():
                create_node_from_bondlist(child_group)

        if self.parent_node_id is not None:
            assert self.parent_node_id in self.node

        self.validate()

        return self




def bond_to_properties(bond):
    this_bond_data = {'direction': bond.GetBondDir(),
                      'stereo': bond.GetStereo()}
    return this_bond_data


def properties_to_bond(mol, atoms, bond_type, props):
    mol.AddBond(atoms[0], atoms[1], bond_type)  # sadly this doesn't return the index of the new bond
    bond = [bond for bond in mol.GetBonds()
            if bond.GetBeginAtomIdx() == atoms[0] and
            bond.GetEndAtomIdx() == atoms[1]][0]
    if 'stereo' in props:
        bond.SetStereo(props['stereo'])
    if 'direction' in props:
        bond.SetBondDir(props['direction'])
    return bond


def atom_to_properties(atom):
    this_node_data = {'atomic_num': atom.GetAtomicNum(),
                      'formal_charge': atom.GetFormalCharge(),
                      'chiral_tag': atom.GetChiralTag(),
                      # 'hybridization': atom.GetHybridization(),
                      # 'num_explicit_hs': atom.GetNumExplicitHs(),
                      # 'is_aromatic':atom.GetIsAromatic(),
                      }
    return this_node_data

def atom_from_properties(props):
    a = Chem.Atom(props['atomic_num'])
    a.SetChiralTag(props['chiral_tag'])
    a.SetFormalCharge(props['formal_charge'])
    # a.SetIsAromatic(node['is_aromatic'])
    # a.SetHybridization(node['hybridization'])
    # a.SetNumExplicitHs(node['num_explicit_hs'])
    return a

def to_mol(graph):
    '''
    Creates an rdkit Molecule object from either a HyperGraphFragment, or a networkx graph exported from it
    :param graph:
    :return:
    '''
    mol = Chem.RWMol()
    node_to_idx = {}
    for node_id, node in graph.node.items():
        if 'dict' in str(type(node)).lower():
            node = node['node']
        a = atom_from_properties(node.data)
        idx = mol.AddAtom(a)
        node_to_idx[node_id] = idx

    for edge_id, edge in graph.edges.items():
        if type(edge_id) == tuple:  # nx.Graph, edge uniquely identified by vertices
            atom_ids = edge_id
            edge = edge['data']
        else:  # HyperGraphFragment, edge has own id
            atom_ids = graph.node_ids_from_edge_id(edge_id)
        if atom_ids[0] not in graph.node or atom_ids[1] not in graph.node:
            continue  # ignore loose edges, if any

        bond = properties_to_bond(mol,
                                  atoms=[node_to_idx[x] for x in atom_ids],
                                  bond_type=edge.type,
                                  props=edge.data)
    RemoveHs(mol)

    Chem.SanitizeMol(mol)
    return mol


if __name__ == '__main__':
    import rdkit

    # TODO: move the below to unit tests!
    smiles = '[CH2+][C@H](O)c1ccccc1\C(F)=C([Br])\C=C'  # 'O=C=O'
    smiles = MolToSmiles(MolFromSmiles(smiles))
    print(smiles)
    mol = MolFromSmiles(smiles)
    hg = HyperGraphFragment.from_mol(mol)
    re_mol = RemoveHs(to_mol(hg))
    re_smiles = MolToSmiles(re_mol)
    print(re_smiles)

    graph = hg.to_nx()
    print(MolToSmiles(RemoveHs(to_mol(graph))))

    # shorthand for grammar
    # rules = {
    #     ('valence_2',['1='])
    #     ('valence_2', [1, 2]): [('valence_4', [1, 2, 3, 4]), ('valence_1', [3]), ('valence_1', [4])],
    #     ('valence_2', [1, 2]): [('valence_4', [1, 2, '3=']), ('valence_2', ['3='])],
    #     ('valence_2', [1, 2]): [('valence_3', [1, 2, 3]), ('valence_1', [3])],
    #     ('valence_3', [1, 2, 3]): [('valence_4', [1, 2, 3, 4]), ('valence_1', [4])]
    # }
    print('done!')
