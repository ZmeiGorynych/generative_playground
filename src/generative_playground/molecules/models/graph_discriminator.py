import torch.nn as nn
from collections import OrderedDict
from generative_playground.codec.hypergraph import HyperGraph
from generative_playground.codec.hypergraph_grammar import GrammarInitializer
from generative_playground.models.embedder.graph_embedder import GraphEmbedder
from generative_playground.molecules.data_utils.zinc_utils import get_zinc_molecules
from generative_playground.models.decoder.graph_decoder import GraphEncoder
from generative_playground.codec.codec import get_codec
from generative_playground.models.heads.attention_aggregating_head import *
from generative_playground.models.heads.multiple_output_head import MultipleOutputHead

class GraphDiscriminator(nn.Module):
    def __init__(self, grammar, drop_rate=0.0):
        super().__init__()
        encoder = GraphEncoder(grammar=grammar,
                       d_model=512,
                       drop_rate=drop_rate)
        encoder_aggregated = FirstSequenceElementHead(encoder)
        self.discriminator = MultipleOutputHead(encoder_aggregated,
                                                {'p_zinc': 2},
                                                drop_rate=drop_rate).to(device)

    def forward(self, x):
        if type(x) in (list, tuple):
            smiles = x
        elif type(x) in (dict, OrderedDict):
            smiles = x['smiles']
        else:
            raise ValueError("Unknown input type: " + str(x))

        mol_graphs = [HyperGraph.from_smiles(s) for s in smiles]
        out = self.discriminator(mol_graphs)
        if type(x) in (list, tuple):
            out['smiles'] = smiles
        elif type(x) in (dict, OrderedDict):
            out.update(x)

        return out
