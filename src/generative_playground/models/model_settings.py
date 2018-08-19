import os, inspect

import generative_playground.models.heads.vae
from generative_playground.codec.character_codec import CharacterModel
from generative_playground.codec.grammar_codec import GrammarModel, zinc_tokenizer, eq_tokenizer, zinc_tokenizer_new
from generative_playground.codec.grammar_helper import grammar_eq, grammar_zinc, grammar_zinc_new
from generative_playground.codec.grammar_mask_gen import GrammarMaskGenerator
from generative_playground.codec.mask_gen_new_2 import GrammarMaskGeneratorNew
from generative_playground.models.decoder.rnn import SimpleRNNDecoder, ResettingRNNDecoder
from generative_playground.models.decoder.resnet_rnn import ResNetRNNDecoder
from generative_playground.models.decoder.random import RandomDecoder
from generative_playground.models.encoder.basic_rnn import SimpleRNN
from generative_playground.models.heads.attention_aggregating_head import AttentionAggregatingHead
from generative_playground.models.encoder.basic_cnn import SimpleCNNEncoder
from generative_playground.models.decoder.decoders import OneStepDecoderContinuous, \
    SimpleDiscreteDecoderWithEnv
from generative_playground.models.heads.masking_head import MaskingHead
from generative_playground.models.decoder.policy import SoftmaxRandomSamplePolicy
from transformer.OneStepAttentionDecoder import SelfAttentionDecoderStep
from transformer.Models import Encoder
from generative_playground.utils.gpu_utils import to_gpu
# in the desired end state, this file will contain every single difference between the different codec

root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
root_location = root_location + '/../../generative_playground/'

eq_charlist = ['x', '+', '(', ')', '1', '2', '3', '*', '/', 's', 'i', 'n', 'e', 'p', ' ']
zinc_charlist =  ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[',
                             '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/',
                             '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']


def get_data_location(molecules=True):
    if molecules:
        return {'source_data': root_location + 'data/250k_rndm_zinc_drugs_clean.smi'}
    else:
        return {'source_data': root_location + 'data/equation2_15_dataset.txt'}


def get_settings(molecules=True, grammar = True):
    data_location = get_data_location(molecules)
    if molecules:
        if grammar == True:
            settings = {'source_data': data_location['source_data'],
                        'data_path': root_location + 'data/zinc_grammar_dataset.h5',
                        'filename_stub': 'grammar_zinc_',
                        'grammar': grammar_zinc,
                        'tokenizer': zinc_tokenizer,
                        'z_size': 56,
                        'decoder_hidden_n': 501, #mkusner/grammarVAE has 501 but that eats too much GPU :)
                        'feature_len': len(grammar_zinc.GCFG.productions()),
                        'max_seq_length': 277,
                        'cnn_encoder_params':{'kernel_sizes': (9, 9, 11),
                                              'filters': (9, 9, 10),
                                              'dense_size': 435},
                        'rnn_encoder_hidden_n': 200,
                        'EPOCHS': 100,
                        'BATCH_SIZE': 300
                        }
        elif grammar == 'new':
            settings = {'source_data': data_location['source_data'],
                        'data_path': root_location + 'data/zinc_grammar_dataset_new.h5',
                        'filename_stub': 'grammar_zinc_',
                        'grammar': grammar_zinc_new,
                        'tokenizer': zinc_tokenizer_new,
                        'z_size': 56,
                        'decoder_hidden_n': 501,  # mkusner/grammarVAE has 501 but that eats too much GPU :)
                        'feature_len': len(grammar_zinc_new.GCFG.productions()),
                        'max_seq_length': 277,
                        'cnn_encoder_params': {'kernel_sizes': (9, 9, 11),
                                               'filters': (9, 9, 10),
                                               'dense_size': 435},
                        'rnn_encoder_hidden_n': 200,
                        'EPOCHS': 100,
                        'BATCH_SIZE': 300
                        }
        else:
            #from generative_playground.codec.character_ed_models import ZincCharacterModel as ThisModel
            settings = {'source_data': data_location['source_data'],
                        'data_path': root_location + 'data/zinc_str_dataset.h5',
                        'filename_stub': 'char_zinc_',
                        'charlist': zinc_charlist,
                        'grammar': None,
                        'z_size': 292,
                        'decoder_hidden_n': 501, #mkusner/grammarVAE has 501 but that eats too much GPU :)
                        'feature_len': len(zinc_charlist),
                        'max_seq_length': 120,
                        'cnn_encoder_params':{'kernel_sizes': (9, 9, 11),
                                              'filters': (9, 9, 10),
                                              'dense_size': 435},

                        'rnn_encoder_hidden_n': 200,
                        'EPOCHS': 100,
                        'BATCH_SIZE': 500
                        }
    else:
        if grammar:
            settings = {'source_data': data_location['source_data'],
                        'data_path':  root_location + 'data/eq2_grammar_dataset.h5',
                        'filename_stub': 'grammar_eq_',
                        'grammar': grammar_eq,
                        'tokenizer': eq_tokenizer,
                        'z_size': 25,
                        'decoder_hidden_n': 100,
                        'feature_len': len(grammar_eq.GCFG.productions()),
                        'max_seq_length': 15,
                        'cnn_encoder_params':{'kernel_sizes': (2, 3, 4),
                                              'filters': (2, 3, 4),
                                              'dense_size': 100},
                        'rnn_encoder_hidden_n': 100,
                        'EPOCHS': 50,
                        'BATCH_SIZE':600
                        }
        else:
            settings = {'source_data': data_location['source_data'],
                        'data_path':  root_location + 'data/eq2_str_dataset.h5',
                        'filename_stub': 'char_eq_',
                        'charlist': eq_charlist,
                        'grammar': None,
                        'z_size': 25,
                        'decoder_hidden_n': 100,
                        'feature_len': len(eq_charlist),
                        'max_seq_length': 31,# max([len(l) for l in L]) L loaded from textfile
                        'cnn_encoder_params': {'kernel_sizes': (2, 3, 4),
                                               'filters': (2, 3, 4),
                                               'dense_size': 100},
                        'rnn_encoder_hidden_n': 100,
                        'EPOCHS': 50,
                        'BATCH_SIZE': 600
                        }
    if grammar:
        settings['codec'] = GrammarModel(max_len=settings['max_seq_length'],
                                         grammar = settings['grammar'],
                                         tokenizer=settings['tokenizer'])
    else:
        settings['codec'] = CharacterModel(max_len=settings['max_seq_length'],
                                           charlist=settings['charlist'])


    return settings

def get_model_args(molecules, grammar,
                   drop_rate=0.5,
                   sample_z = False,
                   encoder_type ='rnn'):

    settings = get_settings(molecules,grammar)
    model_args = {'z_size': settings['z_size'],
                  'decoder_hidden_n':  settings['decoder_hidden_n'],
                  'feature_len': settings['feature_len'],
                  'max_seq_length': settings['max_seq_length'],
                  'cnn_encoder_params':  settings['cnn_encoder_params'],
                  'drop_rate': drop_rate,
                  'sample_z': sample_z,
                  'encoder_type': encoder_type,
                  'rnn_encoder_hidden_n': settings['rnn_encoder_hidden_n']}

    return model_args


def get_model(molecules=True,
              grammar = True,
              weights_file=None,
              epsilon_std=1,
              decoder_type='step',
              **kwargs):
    model_args = get_model_args(molecules=molecules, grammar=grammar)
    for key, value in kwargs.items():
        if key in model_args:
            model_args[key] = value
    sample_z = model_args.pop('sample_z')

    encoder_args = ['feature_len',
                    'max_seq_length',
                    'cnn_encoder_params',
                    'drop_rate',
                    'encoder_type',
                    'rnn_encoder_hidden_n']
    encoder = get_encoder(**{key:value for key, value in model_args.items()
                             if key in encoder_args})

    decoder_args = ['z_size','decoder_hidden_n','feature_len','max_seq_length','drop_rate','batch_size']
    decoder, _ = get_decoder(molecules,
                             grammar,
                             decoder_type=decoder_type,
                             **{key:value for key, value in model_args.items()
                                if key in decoder_args}
                             )

    model = generative_playground.models.heads.vae.VariationalAutoEncoderHead(encoder=encoder,
                                                                              decoder=decoder,
                                                                              sample_z=sample_z,
                                                                              epsilon_std=epsilon_std,
                                                                              z_size=model_args['z_size'])

    if weights_file is not None:
        model.load(weights_file)

    settings = get_settings(molecules=molecules, grammar=grammar)
    if grammar:
        wrapper_model = GrammarModel(max_len=settings['max_seq_length'],
                                 grammar=settings['grammar'],
                                 tokenizer=settings['tokenizer'],#zinc_tokenizer if molecules else eq_tokenizer,
                                 model=model)
    else:
        wrapper_model=CharacterModel(max_len=settings['max_seq_length'],
                                     charlist=settings['charlist'],
                                     model=model
                                     )
    return model, wrapper_model


def get_encoder(feature_len=12,
                max_seq_length=15,
                cnn_encoder_params={'kernel_sizes': (2, 3, 4),
                                               'filters': (2, 3, 4),
                                               'dense_size': 100},
                drop_rate = 0.0,
                encoder_type ='cnn',
                rnn_encoder_hidden_n = 200):

    if encoder_type == 'rnn':
        rnn_model = SimpleRNN(hidden_n=rnn_encoder_hidden_n,
                                           feature_len=feature_len,
                                           drop_rate=drop_rate)
        encoder = to_gpu(AttentionAggregatingHead(rnn_model, drop_rate=drop_rate))

    elif encoder_type == 'cnn':
        encoder = to_gpu(SimpleCNNEncoder(params=cnn_encoder_params,
                                               max_seq_length=max_seq_length,
                                               feature_len=feature_len,
                                               drop_rate=drop_rate))
    elif encoder_type == 'attention':
        encoder = to_gpu(AttentionAggregatingHead(Encoder(feature_len,
                                                          max_seq_length,
                                                          dropout=drop_rate,
                                                          padding_idx=feature_len-1),
                                                  drop_rate=drop_rate))

    else:
        raise NotImplementedError()

    return encoder

def get_decoder(molecules = True,
                        grammar=True,
                        z_size=200,
                 decoder_hidden_n=200,
                 feature_len=12,
                 max_seq_length=15,
                 drop_rate = 0.0,
                        decoder_type='step',
                task = None,
                batch_size=None):
    settings = get_settings(molecules,grammar)
    if decoder_type=='old':
        pre_decoder = ResettingRNNDecoder(z_size=z_size,
                                           hidden_n=decoder_hidden_n,
                                           feature_len=feature_len,
                                           max_seq_length=max_seq_length,
                                          steps=max_seq_length,
                                           drop_rate=drop_rate)
        stepper = OneStepDecoderContinuous(pre_decoder)
    else:
        if decoder_type=='step':
            pre_decoder = SimpleRNNDecoder(z_size=z_size,
                                              hidden_n=decoder_hidden_n,
                                              feature_len=feature_len,
                                              max_seq_length=max_seq_length, #TODO: WHY???
                                              drop_rate=drop_rate,
                                           use_last_action=False)

        elif decoder_type == 'action':
            pre_decoder = SimpleRNNDecoder(z_size=z_size,  # + feature_len,
                                       hidden_n=decoder_hidden_n,
                                       feature_len=feature_len,
                                       max_seq_length=max_seq_length,
                                       drop_rate=drop_rate)

        elif decoder_type == 'action_resnet':
            pre_decoder = ResNetRNNDecoder(z_size=z_size,  # + feature_len,
                                           hidden_n=decoder_hidden_n,
                                           feature_len=feature_len,
                                           max_seq_length=max_seq_length,
                                           drop_rate=drop_rate,
                                           use_last_action=True)

        elif decoder_type == 'attention':
            pre_decoder = SelfAttentionDecoderStep(num_actions=feature_len,
                                                   max_seq_len=max_seq_length,
                                                   drop_rate=drop_rate,
                                                   enc_output_size=z_size)
        elif decoder_type == 'random':
            pre_decoder = RandomDecoder(feature_len=feature_len,
                                        max_seq_length=max_seq_length
                                        )
        else:
            raise NotImplementedError('Unknown decoder type: ' + str(decoder_type))

        stepper = pre_decoder

    if grammar:
        mask_gen = GrammarMaskGeneratorNew(max_seq_length, grammar_zinc_new)#GrammarMaskGenerator(max_seq_length, grammar=settings['grammar'])
        stepper = MaskingHead(stepper, mask_gen)

    policy = SoftmaxRandomSamplePolicy()

    decoder = to_gpu(SimpleDiscreteDecoderWithEnv(stepper,
                                                  policy,
                                                  task=task,
                                                  batch_size=batch_size))#, bypass_actions=True))

    return decoder, pre_decoder


if __name__=='__main__':
    for molecules in [True, False]:
        for grammar in [True,False]:
            wrapper_model = get_model(molecules=molecules, grammar=grammar)

    print('success!')