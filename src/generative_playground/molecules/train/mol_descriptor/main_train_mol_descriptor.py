import os, inspect
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import numpy as np
from random import randint
import torch.nn as nn

from generative_playground.models.decoder.policy import PolicyFromTarget

from generative_playground.models.losses.variational_log_loss import VariationalLoss
from generative_playground.utils.fit import fit
from generative_playground.data_utils.data_sources import MultiDatasetFromHDF5, train_valid_loaders, IterableTransform, \
    SamplingWrapper
from generative_playground.utils.gpu_utils import use_gpu, to_gpu
from generative_playground.molecules.model_settings import get_settings, get_encoder, get_decoder
from generative_playground.metrics.metric_monitor import MetricPlotter
from generative_playground.utils.checkpointer import Checkpointer
from generative_playground.models.heads.mean_variance_head import MeanVarianceSkewHead
from generative_playground.molecules.rdkit_utils.rdkit_utils  import property_scorer
from generative_playground.data_utils.mixed_loader import CombinedLoader
from generative_playground.models.heads.attention_aggregating_head import AttentionAggregatingHead


def train_mol_descriptor(grammar = True,
              EPOCHS = None,
              BATCH_SIZE = None,
              lr = 2e-4,
                         gradient_clip=5,
              drop_rate = 0.0,
              plot_ignore_initial = 0,
              save_file = None,
              preload_file = None,
              encoder_type='rnn',
              plot_prefix = '',
              dashboard='properties',
                aux_dataset=None,
              preload_weights=False):

    root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_location = root_location + '/../'
    save_path = root_location + 'pretrained/' + save_file

    if preload_file is None:
        preload_path = save_path
    else:
        preload_path = root_location + 'pretrained/' + preload_file

    batch_mult = 1 if aux_dataset is None else 2

    settings = get_settings(molecules=True, grammar=grammar)
    max_steps = settings['max_seq_length']

    if EPOCHS is not None:
        settings['EPOCHS'] = EPOCHS
    if BATCH_SIZE is not None:
        settings['BATCH_SIZE'] = BATCH_SIZE
    if False:
        pre_model, _ = get_decoder(True,
                                   grammar,
                                   z_size=settings['z_size'],
                                   decoder_hidden_n=200,
                                   feature_len=settings['feature_len'],
                                   max_seq_length=max_steps,
                                   drop_rate=drop_rate,
                                   decoder_type=encoder_type,
                                   batch_size=BATCH_SIZE*batch_mult)

        class AttentionSimulator(nn.Module):

            def __init__(self, pre_model, drop_rate):
                super().__init__()
                self.pre_model = pre_model
                pre_model_2 = AttentionAggregatingHead(pre_model, drop_rate=drop_rate)
                pre_model_2.model_out_transform = lambda x: x[1]
                self.model = MeanVarianceSkewHead(pre_model_2, 4, drop_rate=drop_rate)

            def forward(self, x):
                self.pre_model.policy = PolicyFromTarget(x)
                return self.model(None)

        model = to_gpu(AttentionSimulator(pre_model, drop_rate=drop_rate))
    else:
        pre_model = get_encoder(feature_len=settings['feature_len'],
                            max_seq_length=settings['max_seq_length'],
                            cnn_encoder_params={'kernel_sizes': (2, 3, 4),
                                                'filters': (2, 3, 4),
                                                'dense_size': 100},
                            drop_rate=drop_rate,
                            encoder_type=encoder_type
                            )

        model = MeanVarianceSkewHead(pre_model, 4, drop_rate=drop_rate)

    nice_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(nice_params, lr=lr)



    main_dataset = MultiDatasetFromHDF5(settings['data_path'],['actions','smiles'])
    train_loader, valid_loader = train_valid_loaders(main_dataset,
                                                     valid_fraction=0.1,
                                                     batch_size=BATCH_SIZE,
                                                     pin_memory=use_gpu)
    def scoring_fun(x):
        if isinstance(x, tuple) or isinstance(x, list):
            x = {'actions': x[0], 'smiles': x[1]}
        out_x = to_gpu(x['actions'])
        end_of_slice = randint(3, out_x.size()[1])
        #TODO inject random slicing back
        out_x = out_x[:, 0:end_of_slice]
        smiles = x['smiles']
        scores = to_gpu(torch.from_numpy(property_scorer(smiles).astype(np.float32)))
        return out_x, scores

    train_gen_main = IterableTransform(train_loader, scoring_fun)
    valid_gen_main = IterableTransform(valid_loader, scoring_fun)

    if aux_dataset is not None:
        train_aux, valid_aux = SamplingWrapper(aux_dataset) \
            .get_train_valid_loaders(BATCH_SIZE,
                                     dataset_name=['actions',
                                                   'smiles'])
        train_gen_aux = IterableTransform(train_aux, scoring_fun)
        valid_gen_aux = IterableTransform(valid_aux, scoring_fun)
        train_gen = CombinedLoader([train_gen_main, train_gen_aux], num_batches=90)
        valid_gen = CombinedLoader([valid_gen_main, valid_gen_aux], num_batches=10)
    else:
        train_gen = train_gen_main#CombinedLoader([train_gen_main, train_gen_aux], num_batches=90)
        valid_gen = valid_gen_main#CombinedLoader([valid_gen_main, valid_gen_aux], num_batches=10)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               factor=0.2,
                                               patience=3,
                                               min_lr=min(0.0001,0.1*lr),
                                               eps=1e-08)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    loss_obj = VariationalLoss(['valid', 'logP', 'SA', 'cyc_sc'])

    metric_monitor = MetricPlotter(plot_prefix=plot_prefix,
                                   loss_display_cap=4.0,
                                   dashboard_name=dashboard,
                                   plot_ignore_initial=plot_ignore_initial)

    checkpointer = Checkpointer(valid_batches_to_checkpoint=10,
                             save_path=save_path)

    fitter = fit(train_gen=train_gen,
                 valid_gen=valid_gen,
                 model=model,
                 optimizer=optimizer,
                 scheduler=scheduler,
                 grad_clip=gradient_clip,
                 epochs=settings['EPOCHS'],
                 loss_fn=loss_obj,
                 metric_monitor = metric_monitor,
                 checkpointer = checkpointer)

    return model, fitter, main_dataset


