import os, inspect
import torch.optim as optim
from torch.optim import lr_scheduler

from generative_playground.models.losses.vae_loss import VAELoss
from generative_playground.utils.fit import fit
from generative_playground.data_utils.data_sources import DatasetFromHDF5, train_valid_loaders, TwinGenerator
from generative_playground.utils.gpu_utils import use_gpu
from generative_playground.molecules.model_settings import get_settings
from generative_playground.models.heads.vae import get_vae
from generative_playground.metrics.metric_monitor import MetricPlotter
from generative_playground.utils.checkpointer import Checkpointer
def train_vae(molecules = True,
              grammar = True,
              EPOCHS = None,
              BATCH_SIZE = None,
              lr = 2e-4,
              drop_rate = 0.0,
              plot_ignore_initial = 0,
              reg_weight = 1,
              epsilon_std = 0.01,
              sample_z = True,
              save_file = None,
              preload_file = None,
              encoder_type='cnn',
              decoder_type='step',
              plot_prefix = '',
              dashboard = 'main',
              preload_weights=False):

    root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_location = root_location + '/../'
    save_path = root_location + 'pretrained/' + save_file
    if preload_file is None:
        preload_path = save_path
    else:
        preload_path = root_location + 'pretrained/' + preload_file



    settings = get_settings(molecules=molecules,grammar=grammar)

    if EPOCHS is not None:
        settings['EPOCHS'] = EPOCHS
    if BATCH_SIZE is not None:
        settings['BATCH_SIZE'] = BATCH_SIZE


    model,_ = get_vae(molecules=molecules,
                      grammar=grammar,
                      drop_rate=drop_rate,
                      sample_z = sample_z,
                      rnn_encoder=encoder_type,
                      decoder_type = decoder_type,
                      weights_file=preload_path if preload_weights else None,
                      epsilon_std=epsilon_std
                      )

    nice_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(nice_params, lr=lr)

    main_dataset = DatasetFromHDF5(settings['data_path'],'data')
    train_loader, valid_loader = train_valid_loaders(main_dataset,
                                                     valid_fraction=0.1,
                                                     batch_size=BATCH_SIZE,
                                                     pin_memory=use_gpu)

    train_gen = TwinGenerator(train_loader)
    valid_gen = TwinGenerator(valid_loader)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               factor=0.2,
                                               patience=3,
                                               min_lr=min(0.0001,0.1*lr),
                                               eps=1e-08)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    loss_obj = VAELoss(settings['grammar'], sample_z, reg_weight)

    metric_monitor = MetricPlotter(plot_prefix=plot_prefix,
                                   loss_display_cap=4.0,
                                   dashboard_name=dashboard,
                                   plot_ignore_initial=plot_ignore_initial)

    checkpointer = Checkpointer(valid_batches_to_checkpoint=1,
                             save_path=save_path)

    fitter = fit(train_gen=train_gen,
                 valid_gen=valid_gen,
                 model=model,
                 optimizer=optimizer,
                 scheduler=scheduler,
                 epochs=settings['EPOCHS'],
                 loss_fn=loss_obj,
                 metric_monitor = metric_monitor,
                 checkpointer = checkpointer)

    return model, fitter, main_dataset


