import os, inspect
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np


from generative_playground.models.losses.variational_log_loss import VariationalLoss
from generative_playground.utils.fit import fit
from generative_playground.data_utils.data_sources import MultiDatasetFromHDF5, train_valid_loaders, IterableTransform
from generative_playground.utils.gpu_utils import use_gpu, to_gpu
from generative_playground.models.model_settings import get_settings, get_encoder
from generative_playground.utils.metric_monitor import MetricPlotter
from generative_playground.utils.checkpointer import Checkpointer
from generative_playground.models.heads.mean_variance_head import MeanVarianceHead
from generative_playground.rdkit_utils.rdkit_utils import property_scorer

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
              dashboard = 'properties',
              preload_weights=False):

    root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_location = root_location + '/../'
    save_path = root_location + 'pretrained/' + save_file
    if preload_file is None:
        preload_path = save_path
    else:
        preload_path = root_location + 'pretrained/' + preload_file



    settings = get_settings(molecules=True,grammar=grammar)

    if EPOCHS is not None:
        settings['EPOCHS'] = EPOCHS
    if BATCH_SIZE is not None:
        settings['BATCH_SIZE'] = BATCH_SIZE


    pre_model = get_encoder(feature_len=settings['feature_len'],
                        max_seq_length=settings['max_seq_length'],
                        cnn_encoder_params={'kernel_sizes': (2, 3, 4),
                                            'filters': (2, 3, 4),
                                            'dense_size': 100},
                        drop_rate=drop_rate,
                        encoder_type=encoder_type
                        )

    model = MeanVarianceHead(pre_model, 4, drop_rate=drop_rate)

    nice_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(nice_params, lr=lr)

    main_dataset = MultiDatasetFromHDF5(settings['data_path'],['data','smiles'])


    train_loader, valid_loader = train_valid_loaders(main_dataset,
                                                     valid_fraction=0.1,
                                                     batch_size=BATCH_SIZE,
                                                     pin_memory=use_gpu)

    def scoring_fun(x):
        out_x = to_gpu(x['data'].float())
        smiles = x['smiles']
        scores = property_scorer(smiles).astype(np.float32)
        return out_x, scores


    train_gen = IterableTransform(train_loader, scoring_fun)
    valid_gen = IterableTransform(valid_loader, scoring_fun)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               factor=0.2,
                                               patience=3,
                                               min_lr=min(0.0001,0.1*lr),
                                               eps=1e-08)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    loss_obj = VariationalLoss()

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


