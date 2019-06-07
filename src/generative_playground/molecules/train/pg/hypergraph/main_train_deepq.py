import os, inspect
from collections import deque
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.utils.data import DataLoader
from generative_playground.molecules.molecule_saver_callback import MoleculeSaver
from generative_playground.molecules.train.pg.hypergraph.visualize_molecules import model_process_fun

from generative_playground.utils.fit_rl import fit_rl
from generative_playground.utils.gpu_utils import to_gpu
from generative_playground.molecules.model_settings import get_settings
from generative_playground.metrics.metric_monitor import MetricPlotter
from generative_playground.utils.checkpointer import Checkpointer
from generative_playground.models.problem.rl.task import SequenceGenerationTask
from generative_playground.models.decoder.decoder import get_decoder, get_node_decoder
from generative_playground.models.losses.policy_gradient_loss import PolicyGradientLoss
from generative_playground.models.problem.policy import SoftmaxRandomSamplePolicy
from generative_playground.data_utils.blended_dataset import EvenlyBlendedDataset
from generative_playground.codec.codec import get_codec
from generative_playground.molecules.data_utils.zinc_utils import get_smiles_from_database
from generative_playground.data_utils.data_sources import IterableTransform, GeneratorToIterable
from generative_playground.molecules.models.graph_discriminator import GraphDiscriminator
from generative_playground.utils.gpu_utils import device


def train_deepq(molecules=True,
                grammar=True,
                smiles_source='ZINC',
                EPOCHS=None,
                BATCH_SIZE=None,
                reward_fun_on=None,
                reward_fun_off=None,
                max_steps=277,
                lr_on=2e-4,
                lr_discrim=1e-4,
                discrim_wt=2,
                p_thresh=0.5,
                drop_rate=0.0,
                plot_ignore_initial=0,
                randomize_reward=False,
                save_file_root_name=None,
                reward_sm=0.0,
                preload_file_root_name=None,
                anchor_file=None,
                anchor_weight=0.0,
                decoder_type='attn',
                plot_prefix='',
                dashboard='deepq',
                smiles_save_file=None,
                on_policy_loss_type='best',
                priors='conditional',
                rule_temperature_schedule=lambda x: 0.01,
                eps=0.0,
                half_float=False):
    root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_location = root_location + '/../../'

    def full_path(x):
        return os.path.realpath(root_location + 'pretrained/' + x)

    if save_file_root_name is not None:
        gen_save_file = save_file_root_name + '_gen.h5'
        disc_save_file = save_file_root_name + '_disc.h5'
    if preload_file_root_name is not None:
        gen_preload_file = preload_file_root_name + '_gen.h5'
        disc_preload_file = preload_file_root_name + '_disc.h5'

    settings = get_settings(molecules=molecules, grammar=grammar)
    # codec = get_codec(molecules, grammar, max_steps)
    # discrim_model = GraphDiscriminator(codec.grammar, drop_rate=drop_rate)
    # if False and preload_file_root_name is not None:
    #     try:
    #         preload_path = full_path(disc_preload_file)
    #         discrim_model.load_state_dict(torch.load(preload_path), strict=False)
    #         print('Discriminator weights loaded successfully!')
    #     except Exception as e:
    #         print('failed to load discriminator weights ' + str(e))

    # zinc_data = get_smiles_from_database(source=smiles_source)
    # zinc_set = set(zinc_data)
    # lookbacks = [BATCH_SIZE, 10*BATCH_SIZE, 100*BATCH_SIZE]
    # history_data = [deque(['O'], maxlen=lb) for lb in lookbacks

    rule_policy = SoftmaxRandomSamplePolicy(temperature=torch.tensor(1.0), eps=eps)
    model, stepper = get_node_decoder(grammar,
                                        max_steps,
                                        drop_rate,
                                        decoder_type,
                                        rule_policy,
                                        reward_fun_on,
                                        BATCH_SIZE,
                                        priors)

    if preload_file_root_name is not None:
        try:
            preload_path = full_path(gen_preload_file)
            model.load_state_dict(torch.load(preload_path, map_location='cpu'), strict=False)
            print('Generator weights loaded successfully!')
        except Exception as e:
            print('failed to load generator weights ' + str(e))

    # anchor_model = None
    #
    # from generative_playground.molecules.rdkit_utils.rdkit_utils import NormalizedScorer
    # import numpy as np
    # scorer = NormalizedScorer()

    # construct the loader to feed the discriminator
    # def make_callback(data):
    #     def hc(inputs, model, outputs, loss_fn, loss):
    #         graphs = outputs['graphs']
    #         smiles = [g.to_smiles() for g in graphs]
    #         for s in smiles:  # only store unique instances of molecules so discriminator can't guess on frequency
    #             if s not in data:
    #                 data.append(s)
    #
    #     return hc

    class TemperatureCallback:
        def __init__(self, policy, temperature_function):
            self.policy = policy
            self.counter = 0
            self.temp_fun = temperature_function

        def __call__(self, inputs, model, outputs, loss_fn, loss):
            self.counter += 1
            target_temp = self.temp_fun(self.counter)
            self.policy.set_temperature(target_temp)

    def get_rl_fitter(model,
                      loss_obj,
                      train_gen,
                      save_path,
                      fit_plot_prefix='',
                      model_process_fun=None,
                      lr=None,
                      extra_callbacks=[],
                      loss_display_cap=float('inf'),
                      anchor_model=None,
                      anchor_weight=0
                      ):
        nice_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(nice_params, lr=lr, eps=1e-4)
        #TODO: try out pulsing lr
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)

        if dashboard is not None:
            metric_monitor = MetricPlotter(plot_prefix=fit_plot_prefix,
                                           loss_display_cap=loss_display_cap,
                                           dashboard_name=dashboard,
                                           plot_ignore_initial=plot_ignore_initial,
                                           process_model_fun=model_process_fun,
                                           smooth_weight=reward_sm)
        else:
            metric_monitor = None

        checkpointer = Checkpointer(valid_batches_to_checkpoint=1,
                                    save_path=save_path,
                                    save_always=True)

        fitter = fit_rl(train_gen=train_gen,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epochs=EPOCHS,
                        loss_fn=loss_obj,
                        grad_clip=5,
                        half_float=half_float,
                        anchor_model=anchor_model,
                        anchor_weight=anchor_weight,
                        callbacks=[metric_monitor, checkpointer] + extra_callbacks
                        )

        return fitter

    def my_gen():
        for _ in range(1000):
            yield to_gpu(torch.zeros(BATCH_SIZE, settings['z_size']))

    # the on-policy fitter

    gen_extra_callbacks = []#[make_callback(d) for d in history_data]

    if smiles_save_file is not None:
        smiles_save_path = os.path.realpath(root_location + 'pretrained/' + smiles_save_file)
        gen_extra_callbacks.append(MoleculeSaver(smiles_save_path, gzip=True))
        print('Saved SMILES to {}'.format(smiles_save_file))

    if rule_temperature_schedule is not None:
        gen_extra_callbacks.append(TemperatureCallback(rule_policy, rule_temperature_schedule))

    fitter1 = get_rl_fitter(model,
                            PolicyGradientLoss(on_policy_loss_type),  # last_reward_wgt=reward_sm),
                            GeneratorToIterable(my_gen),
                            full_path(gen_save_file),
                            plot_prefix + 'on-policy',
                            model_process_fun=model_process_fun,
                            lr=lr_on,
                            extra_callbacks=gen_extra_callbacks)
    #
    # # get existing molecule data to add training
    # pre_dataset = EvenlyBlendedDataset(2 * [history_data[0]] + history_data[1:],
    #                                    labels=False)  # a blend of 3 time horizons
    # dataset = EvenlyBlendedDataset([pre_dataset, zinc_data], labels=True)
    # discrim_loader = DataLoader(dataset, shuffle=True, batch_size=50)
    #
    # class MyLoss(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.celoss = nn.CrossEntropyLoss()
    #
    #     def forward(self, x):
    #         # tmp = discriminator_reward_mult(x['smiles'])
    #         # tmp2 = F.softmax(x['p_zinc'], dim=1)[:,1].detach().cpu().numpy()
    #         # import numpy as np
    #         # assert np.max(np.abs(tmp-tmp2)) < 1e-6
    #         return self.celoss(x['p_zinc'].to(device), x['dataset_index'].to(device))
    #
    # fitter2 = get_rl_fitter(discrim_model,
    #                         MyLoss(),
    #                         IterableTransform(discrim_loader,
    #                                           lambda x: {'smiles': x['X'], 'dataset_index': x['dataset_index']}),
    #                         full_path(disc_save_file),
    #                         plot_prefix + ' discriminator',
    #                         lr=lr_discrim,
    #                         model_process_fun=None)
    #
    # def on_policy_gen(fitter, model):
    #     while True:
    #         # model.policy = SoftmaxRandomSamplePolicy()#bias=codec.grammar.get_log_frequencies())
    #         yield next(fitter)

    return model, fitter1 # ,on_policy_gen(fitter1, model)
