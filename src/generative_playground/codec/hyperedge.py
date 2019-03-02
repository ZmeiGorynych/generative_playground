import copy
import networkx as nx
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AddHs, MolToSmiles, RemoveHs, Kekulize

from collections import OrderedDict


# each edge is a FrozenDict with a random ID
# each node is an OrderedDict with value 'edges' containing a list of edges, and optional other values from rdkit, and has a random ID too


class HyperGraphFragment:
    def __init__(self):
        self.node = OrderedDict()
        self.edges = OrderedDict()
        self.open_edges = list()  # just a list of IDs

    def check_validity(self):
        edge_count = {}
        edge_lists = [node['edges'] for node in self.node.values()] + [self.open_edges]
        for node in edge_lists:
            for edge_id in node:
                if edge_id not in edge_count:
                    edge_count[edge_id] = 1
                else:
                    edge_count[edge_id] += 1

        for count in edge_count.values():
            assert count == 2

    def to_nx(self):
        # create a networkx Graph from self, ignoring any open edges
        # the main information that gets lost hereby is the ordering, so stereochemistry
        # self.check_validity()
        # assert len(self.open_edges) == 0
        G = nx.Graph()
        for node_id, node in self.node.items():
            G.add_node(node_id, **node)

        for edge_id, edge in self.edges.items():
            G.add_edge(edge['nodes'][0], edge['nodes'][1], id=edge_id, **edge)
        return G

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
        for atom in mol.GetAtoms():
            assert atom.GetIdx() not in self.node
            self.node[atom.GetIdx()] = {'atomic_num': atom.GetAtomicNum(),
                                        'formal_charge': atom.GetFormalCharge(),
                                        'chiral_tag': atom.GetChiralTag(),
                                        # 'hybridization': atom.GetHybridization(),
                                        # 'num_explicit_hs': atom.GetNumExplicitHs(),
                                        # 'is_aromatic':atom.GetIsAromatic(),
                                        'edges': [bond.GetIdx() for bond in
                                                  atom.GetBonds()]}  # store the way this atom thinks about edge ordering

        for bond in mol.GetBonds():
            # TODO: replace edge['nodes'] with a function to avoid redundancy
            assert bond.GetIdx() not in self.edges
            self.edges[bond.GetIdx()] = {'nodes': [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
                                         'bond_type': bond.GetBondType(),
                                         'direction': bond.GetBondDir(),
                                         'stereo': bond.GetStereo()}

        return self


def to_mol(graph):
    '''
    Creates an rdkit Molecule object from either a HyperGraphFragment, or a networkx graph exported from it
    :param graph:
    :return:
    '''
    mol = Chem.RWMol()
    node_to_idx = {}
    for node_id, node in graph.node.items():
        a = Chem.Atom(node['atomic_num'])
        a.SetChiralTag(node['chiral_tag'])
        a.SetFormalCharge(node['formal_charge'])
        # a.SetIsAromatic(node['is_aromatic'])
        # a.SetHybridization(node['hybridization'])
        # a.SetNumExplicitHs(node['num_explicit_hs'])
        idx = mol.AddAtom(a)
        node_to_idx[node_id] = idx

    for edge_id, edge in graph.edges.items():
        # TODO: should we be iterating over vertices instead, thus guaranteeing no loose edges
        if type(edge_id) == tuple:  # nx.Graph, edge uniquely identified by vertices
            first, second = edge_id
        else:  # HyperGraphFragment, edge has own id
            first, second = edge['nodes']
        if first not in graph.node or second not in graph.node:
            continue  # control for loose edges, if any
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = edge['bond_type']
        bond_idx = mol.AddBond(ifirst, isecond, bond_type)
        bond = [bond for bond in mol.GetBonds() if bond.GetBeginAtomIdx() == ifirst and
                bond.GetEndAtomIdx() == isecond][
            0]  # bond_idx is 'new number of bonds' rather than the index of the new bond
        bond.SetStereo(edge['stereo'])
        bond.SetBondDir(edge['direction'])

    Chem.SanitizeMol(mol)
    return mol


if __name__ == '__main__':
    import rdkit

    # TODO: move the below to unit tests!
    smiles = '[CH2+][C@H](O)c1ccccc1\C(F)=C([Br])\C=C'  # 'O=C=O'
    smiles = MolToSmiles(MolFromSmiles(smiles))
    print(smiles)
    mol = AddHs(MolFromSmiles(smiles))
    hg = HyperGraphFragment.from_mol(mol)
    re_mol = RemoveHs(to_mol(hg))
    re_smiles = MolToSmiles(re_mol)
    print(re_smiles)

    graph = hg.to_nx()
    print(MolToSmiles(RemoveHs(to_mol(graph))))

    # shorthand for grammar
    rules = {
        ('valence_2',['1='])
        ('valence_2', [1, 2]): [('valence_4', [1, 2, 3, 4]), ('valence_1', [3]), ('valence_1', [4])],
        ('valence_2', [1, 2]): [('valence_4', [1, 2, '3=']), ('valence_2', ['3='])],
        ('valence_2', [1, 2]): [('valence_3', [1, 2, 3]), ('valence_1', [3])],
        ('valence_3', [1, 2, 3]): [('valence_4', [1, 2, 3, 4]), ('valence_1', [4])]
    }
    print('done!')
