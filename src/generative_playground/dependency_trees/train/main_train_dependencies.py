import os, inspect
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import ConcatDataset
import torch
import pickle
import gzip

from generative_playground.models.transformer.Models import TransformerEncoder
from generative_playground.utils.fit import fit
from generative_playground.models.losses.multiple_cross_entropy_loss import MultipleCrossEntropyLoss
from generative_playground.utils.gpu_utils import use_gpu, to_gpu
from generative_playground.metrics.metric_monitor import MetricPlotter
from generative_playground.utils.checkpointer import Checkpointer
from generative_playground.data_utils.data_sources import IterableTransform
from generative_playground.models.heads.multiple_output_head import MultipleOutputHead
from generative_playground.models.decoder.encoder_as_decoder import EncoderAsDecoder
from generative_playground.models.heads.vae import VariationalAutoEncoderHead
from generative_playground.models.embedder.embedder import Embedder
from generative_playground.models.embedder.multi_embedder import MultiEmbedder

def train_dependencies(EPOCHS=None,
                       BATCH_SIZE=None,
                       max_steps=None,
                       feature_len = None,
                       lr=2e-4,
                       drop_rate = 0.0,
                       plot_ignore_initial=1000,
                       save_file = None,
                       preload_file = None,
                       meta=None,
                       languages=None,
                       decoder_type='action',
                       use_self_attention=True,
                       vae=True,
                       target_names=['head'],
                       include_predefined_embedding=True,
                       plot_prefix = '',
                       dashboard = 'policy gradient',
                       ignore_padding=True):

    root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_location = root_location + '/../'
    if save_file is not None:
        save_path = root_location + 'pretrained/' + save_file
    else:
        save_path = None

    settings = {}#get_settings(molecules=molecules,grammar=grammar)

    if EPOCHS is not None:
        settings['EPOCHS'] = EPOCHS
    if BATCH_SIZE is not None:
        settings['BATCH_SIZE'] = BATCH_SIZE

    # task = SequenceGenerationTask(molecules=molecules,
    #                               grammar=grammar,
    #                               reward_fun=reward_fun_on,
    #                               batch_size=BATCH_SIZE,
    #                               max_steps=max_steps,
    #                               save_dataset=save_dataset)

    n_src_vocab = meta['num_tokens'] + 1 # TODO: remmove the +1 after next ingest # the same for all languages by construction
    d_model = 512
    if languages is not None:
        multi_embedder = MultiEmbedder(languages, meta['predefined'], n_src_vocab, d_model)
    else:
        multi_embedder = None
    embedder1 = Embedder(max_steps,
                         n_src_vocab,  # feature_len
                         encode_position=True,
                         include_learned=True,
                         include_predefined=include_predefined_embedding,
                         float_input=False,
                         custom_embedder=multi_embedder
                 )
    encoder = TransformerEncoder(n_src_vocab,
                                 max_steps,
                                 dropout=drop_rate,
                                 padding_idx=0,
                                 embedder=embedder1,
                                 use_self_attention=use_self_attention,
                                 d_model=d_model)

    z_size = encoder.output_shape[2]



    embedder2 = Embedder(max_steps,
                         z_size,  # feature_len
                         encode_position=True,
                         include_learned=True,
                         include_predefined=False,
                         float_input=True,
                         )

    encoder_2 = TransformerEncoder(z_size,
                                   max_steps,
                                   dropout=drop_rate,
                                   padding_idx=0,
                                   embedder=embedder2,
                                   use_self_attention=use_self_attention,
                                   d_model=d_model)

    decoder = EncoderAsDecoder(encoder_2)

    pre_model_2 = VariationalAutoEncoderHead(encoder=encoder,
                                             decoder=decoder,
                                             z_size=z_size,
                                             return_mu_log_var=False)

    if vae:
        pre_model = pre_model_2
    else:
        pre_model = encoder

    model_outputs = {'head': meta['maxlen'],# head
                    'upos': len(meta['upos']),# part of speech
                    'deprel': len(meta['deprel']) # dependency relationship
                    }
    if languages is not None:
        for i in range(len(languages)):
            model_outputs[str(i+1)] = n_src_vocab# word
        loss = MultipleCrossEntropyLoss(multi_language='token',
                                        ignore_padding=ignore_padding)
    else:
        model_outputs['token'] = n_src_vocab
        loss = MultipleCrossEntropyLoss(ignore_padding=ignore_padding)

    model = MultipleOutputHead(pre_model,
                               model_outputs,
                               drop_rate=drop_rate,)

    model = to_gpu(model)

    if preload_file is not None:
        try:
            preload_path = root_location + 'pretrained/' + preload_file
            model.load_state_dict(torch.load(preload_path))
        except:
            pass

    def model_process_fun(model_out, visdom, n):
        pass

    def get_fitter(model,
                   train_gen,
                   valid_gen,
                   loss_obj,
                   fit_plot_prefix='',
                   model_process_fun=None,
                   lr=None,
                   loss_display_cap=float('inf')
                   ):
        nice_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(nice_params, lr=lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   patience=100)#.StepLR(optimizer, step_size=100, gamma=0.99)

        if dashboard is not None:
            metric_monitor = MetricPlotter(plot_prefix=fit_plot_prefix,
                                       loss_display_cap=loss_display_cap,
                                       dashboard_name=dashboard,
                                       plot_ignore_initial=plot_ignore_initial,
                                       process_model_fun=model_process_fun,
                                           smooth_weight=0.9)
        else:
            metric_monitor = None

        checkpointer = Checkpointer(valid_batches_to_checkpoint=1,
                                    save_path=save_path,
                                    save_always=False)

        fitter = fit(train_gen=train_gen,
                    valid_gen=valid_gen,
                    model = model,
                    optimizer = optimizer,
                    scheduler = scheduler,
                    epochs = EPOCHS,
                    loss_fn = loss_obj,
                    batches_to_valid=4,
                    metric_monitor = metric_monitor,
                    checkpointer = checkpointer)

        return fitter

    # TODO: need to be cleaner about dataset creation
    def get_data_loader(dtype, languages):
        if languages is None:
            languages = ['en']
        all_train_data = []
        for lang in languages:
            print('loading', dtype, lang)
            with gzip.open(meta['files'][lang][dtype], 'rb') as f:
                all_train_data.append(pickle.load(f))
        dataset = ConcatDataset(all_train_data)
        loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   pin_memory=use_gpu)
        return loader

    train_loader = get_data_loader('train', languages)
    valid_loader = get_data_loader('valid', languages)

    def extract_input(x):
        if include_predefined_embedding:
            return (to_gpu(x['token']), to_gpu(x['embed']))
        else:
            return to_gpu(x['token'])

    def nice_loader(loader):
        return IterableTransform(loader,
                                 lambda x: (extract_input(x),
                                            {key: to_gpu(val) for key, val in x.items() if key in target_names}))

    # the on-policy fitter
    fitter1 = get_fitter(model,
                         nice_loader(train_loader),
                         nice_loader(valid_loader),
                         loss,
                         plot_prefix,
                         model_process_fun=model_process_fun,
                         lr=lr)


    return model, fitter1

