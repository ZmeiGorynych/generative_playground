from generative_playground.codec.hypergraph import HyperGraph
from generative_playground.utils.gpu_utils import device
import torch
from torch import nn
from torch.nn import Linear
import networkx as nx


class GraphEmbedder(nn.Module):
    def __init__(self, target_dim, max_nodes=256, grammar=None, embed_self=True):
        '''

        :param max_nodes: maximum number of nodes in a graph
        :param target_dim: output dimension
        :param grammar: HypergraphGrammar, used to encode the node data, if any
        '''
        super().__init__()
        self.embed_self = embed_self
        self.pre_output_shape = [None, None, 2*max_nodes +
                                 (0 if grammar is None else grammar.node_data_index_length())]
        self.output_shape = [None, None, target_dim]
        self.grammar = grammar
        self.embed = Linear(self.pre_output_shape[-1], target_dim).to(device)
        self.max_nodes = max_nodes

    def float_type(self):
        return self.embed.weight.dtype


    def forward(self, graphs):
        """
        Takes a batch of HyperGraphs or nx.Graphs, returns am embedding of their connectivity and node data
        WARNING: nx.Graphs are not ordered, so use this on them only if you don't care about node embedding order
        :param graphs: a list containing the graphs
        :return: float32s of shape len(graphs) x max([len(g) for g in graphs]) x target_dim
        return second dimension is variable to save space if all graphs have a small number of nodes
         All nodes exceeding max_nodes are ignored
        """
        # the very first time, we need to pick the starting node, so all the graphs passed in will be None
        batch_size = len(graphs)
        if all([graph is None for graph in graphs]):
            return 0.1*torch.ones(batch_size, 1, self.output_shape[-1], device=device, dtype=self.float_type())

        # if we got this far, it's not the first step anymore - but we still could get Nones if we sample from the experience buffer
        for graph in graphs:
            assert isinstance(graph, HyperGraph) or isinstance(graph, nx.Graph) or graph is None , \
                "Wrong input type:" + str(
                type(graph))

        this_max_nodes = max([1 if g is None else len(g) for g in graphs])
        # TODO: log if this is > self.max_nodes
        this_pre_output_shape = (batch_size, this_max_nodes, self.pre_output_shape[-1])
        # this_output_shape = (batch_size,  this_max_nodes, self.output_shape[-1])

        out = torch.zeros(*this_pre_output_shape, device=device, dtype=self.float_type())

        for i, graph in enumerate(graphs):
            if graph is None:
                out[i,0,:] = 0.1
            else:
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
                    if idx1 < this_max_nodes and idx2 < this_max_nodes:
                        out[i, idx1, idx2] = wgt
                        out[i, idx2, idx1] = wgt

                # now encode node data
                for n, node in enumerate(graph.node.values()):
                    if n >= self.max_nodes:
                        break
                    if self.embed_self:
                        out[i, n, self.max_nodes + n] = 1
                    if hasattr(node, 'data'):
                        # TODO: also encode the is_terminal values
                        assert n < self.max_nodes, "Graph has too many nodes"
                        offset = self.max_nodes*2
                        for fn in self.grammar.node_data_index.keys():
                            this_dict = self.grammar.node_data_index[fn]
                            if fn in node.data:
                                if node.data[fn] in this_dict:
                                    out[i, n, offset + this_dict[node.data[fn]]] = 1
                                else:
                                    out[i, n, offset + len(this_dict)] = 1 # 'other'
                            offset += len(this_dict) + 1

        out2 = self.embed(out)
        # now nuke all the values not associated with nodes
        for i, g in enumerate(graphs):
            if g is not None:
                out2[i,len(g):,:] = 0

        return out2



