from generative_playground.molecules.lean_settings import root_location, get_data_location


# in the desired end state, this file will contain every single difference between the different codec

# root_location = root_location + '/../../generative_playground/'


def get_settings(molecules=True, grammar=True):
    if grammar is True:
        grammar = 'classic'
    data_location = get_data_location(molecules)
    if molecules:
        settings = {'source_data': data_location['source_data'],
                    'data_path': root_location + 'data/zinc_grammar_dataset.h5',
                    'filename_stub': 'grammar_zinc_' + str(grammar) + '_',
                    'decoder_hidden_n': 501,
                    'cnn_encoder_params': {'kernel_sizes': (9, 9, 11),
                                           'filters': (9, 9, 10),
                                           'dense_size': 435},
                    'rnn_encoder_hidden_n': 200,
                    'EPOCHS': 100,
                    }
        if grammar is False:
            settings.update({'filename_stub': 'char_zinc_',
                        'z_size': 292,
                        'max_seq_length': 120,
                        'BATCH_SIZE': 500
                        })
        elif grammar == 'classic':
            settings.update({'z_size': 56,
                             'max_seq_length': 277,
                             'BATCH_SIZE': 300
                             })
        elif grammar == 'new':
            settings.update({
                        'z_size': 56,
                        'max_seq_length': 277,
                        'BATCH_SIZE': 300
                        })

        elif 'hypergraph' in grammar:
            settings.update({'z_size': 56,
                             'max_seq_length': 277,
                             'BATCH_SIZE': 300
                             })



    else: # equations encoding-decoding
        settings = {'source_data': data_location['source_data'],
                    'data_path': root_location + 'data/eq2_grammar_dataset.h5',
                    'z_size': 25,
                    'decoder_hidden_n': 100,
                    'cnn_encoder_params': {'kernel_sizes': (2, 3, 4),
                                           'filters': (2, 3, 4),
                                           'dense_size': 100},
                    'rnn_encoder_hidden_n': 100,
                    'EPOCHS': 50,
                    'BATCH_SIZE': 600
                    }
        if grammar:
            settings.update({'filename_stub': 'grammar_eq_',
                            'max_seq_length': 15,
                             })
        else:
            settings.update({'filename_stub': 'char_eq_',
                             'max_seq_length': 31,  # max([len(l) for l in L]) L loaded from textfile
                             })

    return settings


