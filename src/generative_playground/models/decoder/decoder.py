from generative_playground.codec.codec import get_codec
from generative_playground.models.decoder.decoders import OneStepDecoderContinuous, SimpleDiscreteDecoderWithEnv
from generative_playground.models.decoder.resnet_rnn import ResNetRNNDecoder
from generative_playground.models.decoder.rnn import ResettingRNNDecoder, SimpleRNNDecoder
from generative_playground.models.decoder.stepper import RandomDecoder
from generative_playground.models.decoder.graph_decoder import *

from generative_playground.models.heads import MultipleOutputHead, MaskingHead
from generative_playground.models.transformer.OneStepAttentionDecoder import SelfAttentionDecoderStep
from generative_playground.utils.gpu_utils import to_gpu
from generative_playground.models.decoder.decoders import DecoderWithEnvironmentNew
from generative_playground.models.problem.rl.environment import GraphEnvironment
from generative_playground.molecules.models.graph_models import get_graph_model

def get_decoder(molecules=True,
                grammar=True,
                z_size=200,
                decoder_hidden_n=200,
                feature_len=12,  # TODO: remove this
                max_seq_length=15,
                drop_rate=0.0,
                decoder_type='step',
                task=None,
                node_policy=None,
                rule_policy=None,
                reward_fun=lambda x: -1 * np.ones(len(x)),
                batch_size=None,
                priors=True):
    codec = get_codec(molecules, grammar, max_seq_length)

    if decoder_type == 'old':
        stepper = ResettingRNNDecoder(z_size=z_size,
                                      hidden_n=decoder_hidden_n,
                                      feature_len=codec.feature_len(),
                                      max_seq_length=max_seq_length,
                                      steps=max_seq_length,
                                      drop_rate=drop_rate)
        stepper = OneStepDecoderContinuous(stepper)
    elif 'graph' in decoder_type and decoder_type not in ['attn_graph', 'rnn_graph']:
        return get_node_decoder(grammar,
                     max_seq_length,
                     drop_rate,
                     decoder_type,
                     rule_policy,
                     reward_fun,
                     batch_size,
                     priors)

    elif decoder_type in ['attn_graph', 'rnn_graph']: # deprecated
        assert 'hypergraph' in grammar, "Only the hypergraph grammar can be used with attn_graph decoder type"
        if 'attn' in decoder_type:
            encoder = GraphEncoder(grammar=codec.grammar,
                                   d_model=512,
                                   drop_rate=drop_rate,
                                   model_type='transformer')
        elif 'rnn' in decoder_type:
            encoder = GraphEncoder(grammar=codec.grammar,
                                   d_model=512,
                                   drop_rate=drop_rate,
                                   model_type='rnn')

        model = MultipleOutputHead(model=encoder,
                                   output_spec={'node': 1,  # to be used to select next node to expand
                                                'action': codec.feature_len()},  # to select the action for chosen node
                                   drop_rate=drop_rate)

        # don't support using this model in VAE-style models yet
        model.init_encoder_output = lambda x: None
        mask_gen = HypergraphMaskGenerator(max_len=max_seq_length,
                                           grammar=codec.grammar)
        mask_gen.priors = priors
        # bias=codec.grammar.get_log_frequencies())
        if node_policy is None:
            node_policy = SoftmaxRandomSamplePolicy()
        if rule_policy is None:
            rule_policy = SoftmaxRandomSamplePolicy()
        if 'node' in decoder_type:
            stepper = GraphDecoderWithNodeSelection(model,
                                                    node_policy=node_policy,
                                                    rule_policy=rule_policy)
            env = GraphEnvironment(mask_gen,
                                   reward_fun=reward_fun,
                                   batch_size=batch_size)
            decoder = DecoderWithEnvironmentNew(stepper, env)
        else:

            stepper = GraphDecoder(model=model, mask_gen=mask_gen)
            decoder = to_gpu(SimpleDiscreteDecoderWithEnv(stepper,
                                                          rule_policy,
                                                          task=task,
                                                          batch_size=batch_size))
        return decoder, stepper

    else:
        if decoder_type == 'step':
            stepper = SimpleRNNDecoder(z_size=z_size,
                                       hidden_n=decoder_hidden_n,
                                       feature_len=codec.feature_len(),
                                       max_seq_length=max_seq_length,
                                       drop_rate=drop_rate,
                                       use_last_action=False)

        elif decoder_type == 'action':
            stepper = SimpleRNNDecoder(z_size=z_size,  # + feature_len,
                                       hidden_n=decoder_hidden_n,
                                       feature_len=codec.feature_len(),
                                       max_seq_length=max_seq_length,
                                       drop_rate=drop_rate,
                                       use_last_action=True)

        elif decoder_type == 'action_resnet':
            stepper = ResNetRNNDecoder(z_size=z_size,  # + feature_len,
                                       hidden_n=decoder_hidden_n,
                                       feature_len=codec.feature_len(),
                                       max_seq_length=max_seq_length,
                                       drop_rate=drop_rate,
                                       use_last_action=True)

        elif decoder_type == 'attention':
            stepper = SelfAttentionDecoderStep(num_actions=codec.feature_len(),
                                               max_seq_len=max_seq_length,
                                               drop_rate=drop_rate,
                                               enc_output_size=z_size)
        elif decoder_type == 'random':
            stepper = RandomDecoder(feature_len=codec.feature_len(),
                                    max_seq_length=max_seq_length
                                    )
        else:
            raise NotImplementedError('Unknown decoder type: ' + str(decoder_type))

    if grammar is not False and '_graph' not in decoder_type:
        # add a masking layer
        mask_gen = get_codec(molecules, grammar, max_seq_length).mask_gen
        stepper = MaskingHead(stepper, mask_gen)

    policy = SoftmaxRandomSamplePolicy()  # bias=codec.grammar.get_log_frequencies())

    decoder = to_gpu(SimpleDiscreteDecoderWithEnv(stepper,
                                                  policy,
                                                  task=task,
                                                  batch_size=batch_size))  # , bypass_actions=True))

    return decoder, stepper


def get_node_decoder(grammar,
                     max_seq_length=15,
                     drop_rate=0.0,
                     decoder_type='attn',
                     rule_policy=None,
                     reward_fun=lambda x: -1 * np.ones(len(x)),
                     batch_size=None,
                     priors='conditional',
                     bins=10):

    codec = get_codec(True, grammar, max_seq_length)
    assert 'hypergraph' in grammar, "Only the hypergraph grammar can be used with attn_graph decoder type"
    if 'attn' in decoder_type:
        model_type='transformer'
    elif 'rnn' in decoder_type:
        model_type = 'rnn'
    elif 'conditional' in decoder_type:
        if 'sparse' in decoder_type:
            model_type = 'conditional_sparse'
        else:
            model_type = 'conditional'

    if 'distr' in decoder_type:
        if 'softmax' in decoder_type:
            output_type = 'distributions_softmax'
        else:
            output_type = 'distributions_thompson'
    else:
        output_type = 'values'

    model = get_graph_model(codec, drop_rate, model_type, output_type, num_bins=bins)

    mask_gen = HypergraphMaskGenerator(max_len=max_seq_length,
                                       grammar=codec.grammar)
    mask_gen.priors = priors
    if rule_policy is None:
        rule_policy = SoftmaxRandomSamplePolicySparse() if 'sparse' in decoder_type else SoftmaxRandomSamplePolicy()

    stepper_type = GraphDecoderWithNodeSelectionSparse if 'sparse' in decoder_type else GraphDecoderWithNodeSelection
    stepper = stepper_type(model,
                                            rule_policy=rule_policy)
    env = GraphEnvironment(mask_gen,
                           reward_fun=reward_fun,
                           batch_size=batch_size)
    decoder = DecoderWithEnvironmentNew(stepper, env)

    return decoder, stepper
