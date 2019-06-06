import numpy as np

from generative_playground.codec.hypergraph_grammar import GrammarInitializer, HypergraphMaskGenerator
from generative_playground.models.decoder.decoders import DecoderWithEnvironmentNew
from generative_playground.models.decoder.graph_decoder import GraphDecoderWithNodeSelection
from generative_playground.models.problem.rl.environment import GraphEnvironment
from generative_playground.molecules.models.graph_discriminator import GraphTransformerModel



def make_grammar():
    tmp_file = 'tmp2.pickle'
    gi = GrammarInitializer(tmp_file)
    gi.delete_cache()
    # now create a clean new one
    gi = GrammarInitializer(tmp_file)
    # run a first run for 10 molecules
    gi.init_grammar(20)
    gi.grammar.check_attributes()
    return gi.grammar


def make_decoder(grammar, output_spec):
    model = GraphTransformerModel(grammar, output_spec, drop_rate=0.0, d_model=512)
    stepper = GraphDecoderWithNodeSelection(model)
    env = make_environment(grammar, batch_size=2)
    decoder = DecoderWithEnvironmentNew(stepper, env)
    return decoder


def make_environment(grammar, batch_size=2):
    mask_gen = HypergraphMaskGenerator(30, grammar, priors='conditional')
    env = GraphEnvironment(mask_gen,
                           reward_fun=lambda x: 2 * np.ones(len(x)),
                           batch_size=batch_size)
    return env