import os, inspect
import torch.optim as optim
from torch.optim import lr_scheduler

from generative_playground.models.problem.variational_autoencoder import VAELoss
from generative_playground.fit import fit
from generative_playground.data_utils.data_sources import DatasetFromHDF5, train_valid_loaders, DuplicateIter
from generative_playground.gpu_utils import use_gpu
from generative_playground.models.model_settings import get_settings, get_model


def train_vae(molecules = True,
              grammar = True,
              EPOCHS = None,
              BATCH_SIZE = None,
              lr = 2e-4,
              drop_rate = 0.0,
              plot_ignore_initial = 0,
              KL_weight = 1,
              epsilon_std = 0.01,
              sample_z = True,
              save_file = None,
              rnn_encoder='cnn',
              decoder_type='step',
              plot_prefix = '',
              dashboard = 'main',
              preload_weights=False):
    root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_location = root_location + '/../'
    save_path = root_location + 'pretrained/' + save_file

    settings = get_settings(molecules=molecules,grammar=grammar)

    if EPOCHS is not None:
        settings['EPOCHS'] = EPOCHS
    if BATCH_SIZE is not None:
        settings['BATCH_SIZE'] = BATCH_SIZE

    # model_args = get_model_args(molecules,
    #                             grammar,
    #                             drop_rate=drop_rate,
    #                             sample_z = sample_z,
    #                             rnn_encoder=rnn_encoder)

    model,_ = get_model(molecules=molecules,
                        grammar=grammar,
                        drop_rate=drop_rate,
                        sample_z = sample_z,
                        rnn_encoder=rnn_encoder,
                        decoder_type = decoder_type,
                        weights_file=save_path if preload_weights else None,
                        epsilon_std=0.01
                        )

    #model = GrammarVariationalAutoEncoder(**model_args)
    # if preload_weights:
    #     try:
    #         model.load(save_path)
    #     except:
    #         pass
    nice_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(nice_params, lr=lr)

    # TODO: create this outside and pass in?
    main_dataset = DatasetFromHDF5(settings['data_path'],'data')
    train_loader, valid_loader = train_valid_loaders(main_dataset,
                                                     valid_fraction=0.1,
                                                     batch_size=BATCH_SIZE,
                                                     pin_memory=use_gpu)

    train_gen = DuplicateIter(train_loader)
    valid_gen = DuplicateIter(valid_loader)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               factor=0.2,
                                               patience=3,
                                               min_lr=0.0001,
                                               eps=1e-08)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    loss_obj = VAELoss(settings['grammar'], sample_z, KL_weight)

    fitter = fit(train_gen=train_gen,
                 valid_gen=valid_gen,
                 model=model,
                 optimizer=optimizer,
                 scheduler=scheduler,
                 epochs=settings['EPOCHS'],
                 loss_fn=loss_obj,
                 save_path=save_path,
                 dashboard_name=dashboard,
                 plot_ignore_initial=plot_ignore_initial,
                 plot_prefix=plot_prefix)

    return model, fitter, main_dataset


