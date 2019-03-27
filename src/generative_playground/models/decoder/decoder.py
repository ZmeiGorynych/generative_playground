from generative_playground.codec.codec import get_codec
from generative_playground.codec.hypergraph_grammar import HypergraphMaskGenerator
from generative_playground.models.decoder.decoders import OneStepDecoderContinuous, SimpleDiscreteDecoderWithEnv
from generative_playground.models.decoder.policy import SoftmaxRandomSamplePolicy
from generative_playground.models.decoder.resnet_rnn import ResNetRNNDecoder
from generative_playground.models.decoder.rnn import ResettingRNNDecoder, SimpleRNNDecoder
from generative_playground.models.decoder.stepper import RandomDecoder
from generative_playground.models.decoder.graph_decoder import *

from generative_playground.models.heads import MultipleOutputHead, MaskingHead
from generative_playground.models.transformer.OneStepAttentionDecoder import SelfAttentionDecoderStep
from generative_playground.molecules.model_settings import get_settings
from generative_playground.utils.gpu_utils import to_gpu


def get_decoder(molecules=True,
                grammar=True,
                z_size=200,
                decoder_hidden_n=200,
                feature_len=12, # TODO: remove this
                max_seq_length=15,
                drop_rate=0.0,
                decoder_type='step',
                task=None,
                batch_size=None):

    codec = get_codec(molecules, grammar, max_seq_length)

    if decoder_type == 'old':
        stepper = ResettingRNNDecoder(z_size=z_size,
                                          hidden_n=decoder_hidden_n,
                                          feature_len=codec.feature_len(),
                                          max_seq_length=max_seq_length,
                                          steps=max_seq_length,
                                          drop_rate=drop_rate)
        stepper = OneStepDecoderContinuous(stepper)
    elif decoder_type == 'attn_graph':
        assert 'hypergraph' in grammar
        encoder = GraphEncoder(grammar=codec.grammar,
                               d_model=512,
                               drop_rate=drop_rate)
        model = MultipleOutputHead(model=encoder,
                                   output_spec={'node':[], 'action':[]},
                                   drop_rate=drop_rate)
        mask_gen = HypergraphMaskGenerator(max_len=max_seq_length,
                                      grammar=codec.grammar)

        stepper = GraphDecoder(model=model,
                               mask_gen=mask_gen,
                               grammar=codec.grammar,
                               node_selection_policy=SoftmaxRandomSamplePolicy())

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

    if grammar is not False:
        # add a masking layer
        mask_gen = get_codec(molecules, grammar, max_seq_length).mask_gen
        stepper = MaskingHead(stepper, mask_gen)

    policy = SoftmaxRandomSamplePolicy()

    decoder = to_gpu(SimpleDiscreteDecoderWithEnv(stepper,
                                                  policy,
                                                  task=task,
                                                  batch_size=batch_size))  # , bypass_actions=True))

    return decoder, stepper