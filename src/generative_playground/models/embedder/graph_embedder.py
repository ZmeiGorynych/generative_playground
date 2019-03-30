from generative_playground.codec.hypergraph import HyperGraph
from generative_playground.utils.gpu_utils import device
import torch
from torch import nn
from torch.nn import Linear
import networkx as nx

class GraphEmbedder(nn.Module):
    def __init__(self, max_nodes, target_dim, grammar=None):
        '''

        :param max_nodes: maximum number of nodes in a graph
        :param grammar: HypergraphGrammar, used to encode the node data, if any
        '''
        super().__init__()
        self.pre_output_shape = [None, max_nodes, max_nodes +
                                 (0 if grammar is None else grammar.node_data_index_length())]
        self.output_shape = [None, max_nodes, target_dim]
        self.grammar = grammar
        self.embed = Linear(self.pre_output_shape[-1], target_dim).to(device)


    def forward(self, graphs):
        """
        Takes a batch of HyperGraphs or nx.Graphs, returns am embedding of their connectivity and node data
        WARNING: nx.Graphs are not ordered, so use this on them only if you don't care about node embedding order
        :param graphs: a list containing the graphs
        :return: float32s of self.output_shape
        """
        # the very first time, we need to pick the starting node, so all the graphs passed in will be None
        batch_size = len(graphs)
        this_pre_output_shape = (batch_size, *self.pre_output_shape[1:])
        this_output_shape = (batch_size, *self.output_shape[1:])

        if all([graph is None for graph in graphs]):
            return 0.1*torch.ones(*this_output_shape, device=device)

        # if we got this far, it's not the first step anymore
        for graph in graphs:
            assert isinstance(graph, HyperGraph) or isinstance(graph, nx.Graph), "Wrong input type:" + str(type(graph))

        out = torch.zeros(*this_pre_output_shape, device=device)

        for i, graph in enumerate(graphs):
            # first encode the connectivity, including bond type (single/double/etc) if available
            node_id_to_idx = {id: idx for idx, id in enumerate(graph.node.keys())}
            for edge_id, edge in graph.edges.items():
                try: # HyperGraph
                    node1_id, node2_id = graph.node_ids_from_edge_id(edge_id)
                except: # nx.Graph
                    node1_id, node2_id = graph.edges[edge_id]

                idx1 = node_id_to_idx[node1_id]
                idx2 = node_id_to_idx[node2_id]
                if hasattr(edge,'type'):
                    wgt = float(edge.type)
                else:
                    wgt = 1

                out[i, idx1, idx2] = wgt
                out[i, idx2, idx1] = wgt

            # now encode node data
            for n, node in enumerate(graph.node.values()):
                if hasattr(node, 'data'):
                    assert n < self.pre_output_shape[1], "Graph has too many nodes"
                    offset = self.pre_output_shape[1]
                    for fn in self.grammar.node_data_index.keys():
                        this_dict = self.grammar.node_data_index[fn]
                        if fn in node.data:
                            if node.data[fn] in this_dict:
                                out[i, n, offset + this_dict[node.data[fn]]] = 1
                            else:
                                out[i, n, offset + len(this_dict)] = 1 # 'other'
                        offset += len(this_dict) + 1

            out2 = self.embed(out.view(-1, self.pre_output_shape[-1]))
            out3 = out2.view(*this_output_shape)
        return out3



