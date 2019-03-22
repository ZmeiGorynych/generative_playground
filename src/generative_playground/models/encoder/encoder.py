from generative_playground.models.encoder.basic_cnn import SimpleCNNEncoder
from generative_playground.models.encoder.basic_rnn import SimpleRNN
from generative_playground.models.heads.attention_aggregating_head import AttentionAggregatingHead
from generative_playground.models.transformer.Models import TransformerEncoder
from generative_playground.utils.gpu_utils import to_gpu


def get_encoder(feature_len=12,
                max_seq_length=15,
                cnn_encoder_params={'kernel_sizes': (2, 3, 4),
                                    'filters': (2, 3, 4),
                                    'dense_size': 100},
                drop_rate=0.0,
                encoder_type='cnn',
                rnn_encoder_hidden_n=200):
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
        encoder = to_gpu(AttentionAggregatingHead(TransformerEncoder(feature_len,
                                                                     max_seq_length,
                                                                     dropout=drop_rate,
                                                                     padding_idx=feature_len - 1),
                                                  drop_rate=drop_rate))

    else:
        raise NotImplementedError()

    return encoder