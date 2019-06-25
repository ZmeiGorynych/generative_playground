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
from generative_playground.models.discrete_distribution_utils import CalcExpectedValue

class GraphDiscriminator(nn.Module):
    def __init__(self, grammar, drop_rate=0.0, d_model=512):
        super().__init__()
        encoder = GraphEncoder(grammar=grammar,
                       d_model=d_model,
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

class GraphTransformerModel(nn.Module):
    def __init__(self, grammar, output_spec, drop_rate=0.0, d_model=512):
        super().__init__()
        encoder = GraphEncoder(grammar=grammar,
                               d_model=d_model,
                               drop_rate=drop_rate)
        self.model = MultipleOutputHead(encoder,
                                                output_spec,
                                                drop_rate=drop_rate).to(device)

        # don't support using this model in VAE-style models yet
        self.init_encoder_output = lambda x: None
        self.output_shape = self.model.output_shape

    def forward(self, mol_graphs):
        """

        :param mol_graphs: graphs: a list of Hypergraphs
        :return: batch x max_nodes x <output_spec> floats
        """
        out = self.model(mol_graphs)
        out['graphs'] = mol_graphs

        return out


def get_graph_model(codec, drop_rate, model_type, output_type='values', num_bins=51):
    encoder = GraphEncoder(grammar=codec.grammar,
                           d_model=512,
                           drop_rate=drop_rate,
                           model_type=model_type)
    if output_type == 'values':
        model = MultipleOutputHead(model=encoder,
                                   output_spec={'node': 1,  # to be used to select next node to expand
                                                'action': codec.feature_len()},  # to select the action for chosen node
                                   drop_rate=drop_rate)
        model = OneValuePerNodeRuleTransform(model)
    elif output_type == 'distributions':
        model = MultipleOutputHead(model=encoder,
                                   output_spec={'node': 1,  # to be used to select next node to expand
                                                'action': codec.feature_len()*num_bins},  # to select the action for chosen node
                                   drop_rate=drop_rate)

        model = DistributionPerNodeRuleTransform(model, num_bins=num_bins, T=0.1)

    model.init_encoder_output = lambda x: None
    return model

class OneValuePerNodeRuleTransform(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.output_shape = model.output_shape

    def handle_priors(self, next_logits, full_logit_priors):
        dtype = next_logits.dtype
        batch_size = len(next_logits)
        full_priors_pytorch = torch.from_numpy(full_logit_priors).to(device=device, dtype=dtype).view(batch_size, -1)
        return next_logits + full_priors_pytorch, full_priors_pytorch

    def forward(self, x):
        pre_out = self.model(x)
        batch_size = len(pre_out['action'])
        pre_out['action_p_logits'] = pre_out['action'].view(batch_size, -1)
        return pre_out

class DistributionPerNodeRuleTransform(nn.Module):
    def __init__(self, model, num_bins, T=1):
        super().__init__()
        self.model = model
        self.num_bins = num_bins
        self.T = T
        self.exp_value = CalcExpectedValue() # TODO: replace with Thompson later
        self.output_shape = model.output_shape # TODO: fix!

    def action_logits_to_action_prob_logits(self, action_logits):
        assert len(action_logits.size()) == 3
        # convert probability logits to probability distributions per action
        action_probs = F.softmax(action_logits, dim=2)
        # and calculate their exp values, later to be replaced by log Thompson probabilities
        action_p_logits = self.exp_value(action_probs/ self.T)
        return {'action_p_logits': action_p_logits, 'action_distrs': action_probs}

    def handle_priors(self, next_logits, full_logit_priors):
        dtype = next_logits.dtype
        batch_size = len(next_logits)
        full_priors_pytorch = torch.from_numpy(full_logit_priors).to(device=device, dtype=dtype).view(batch_size, -1)
        out_logits = next_logits
        out_logits[full_priors_pytorch < -1e3] = -1e5
        return next_logits, 0.0

    def forward(self, x):
        pre_out = self.model(x)
        batch_size = len(pre_out['action'])
        pre_out['action'] = pre_out['action'].view(batch_size, -1, self.num_bins) # normalize to batch x (node*rules) x bins
        extras = self.action_logits_to_action_prob_logits(pre_out['action'])
        pre_out.update(extras)
        return pre_out