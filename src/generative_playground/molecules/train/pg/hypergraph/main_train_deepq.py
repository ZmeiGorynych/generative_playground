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

from generative_playground.utils.fit import fit
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
from generative_playground.models.problem.rl.deepq import QLearningDataset, DeepQModelWrapper, DeepQLoss, collate_experiences


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

    # settings = get_settings(molecules=molecules, grammar=grammar)

    rule_policy = SoftmaxRandomSamplePolicy(temperature=torch.tensor(1.0), eps=eps)
    decoder, stepper = get_node_decoder(grammar,
                                        max_steps,
                                        drop_rate,
                                        decoder_type,
                                        rule_policy,
                                        reward_fun_on,
                                        BATCH_SIZE,
                                        priors)

    decoder.stepper.detach_model_output = True

    if preload_file_root_name is not None:
        try:
            preload_path = full_path(gen_preload_file)
            decoder.load_state_dict(torch.load(preload_path, map_location='cpu'), strict=False)
            print('Generator weights loaded successfully!')
        except Exception as e:
            print('failed to load generator weights ' + str(e))

    class TemperatureCallback:
        def __init__(self, policy, temperature_function):
            self.policy = policy
            self.counter = 0
            self.temp_fun = temperature_function

        def __call__(self, inputs, model, outputs, loss_fn, loss):
            self.counter += 1
            target_temp = self.temp_fun(self.counter)
            self.policy.set_temperature(target_temp)

    def get_fitter(model,
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
                                    save_always=True,
                                    verbose=1)

        fitter = fit(train_gen=train_gen,
                     valid_gen=train_gen,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epochs=EPOCHS,
                        loss_fn=loss_obj,
                        grad_clip=5,
                        #half_float=half_float,
                        # anchor_model=anchor_model,
                        # anchor_weight=anchor_weight,
                        callbacks=[metric_monitor, checkpointer] + extra_callbacks
                        )

        return fitter

    gen_extra_callbacks = []

    if smiles_save_file is not None:
        smiles_save_path = os.path.realpath(root_location + 'pretrained/' + smiles_save_file)
        gen_extra_callbacks.append(MoleculeSaver(smiles_save_path, gzip=True))
        print('Saved SMILES to {}'.format(smiles_save_file))

    if rule_temperature_schedule is not None:
        gen_extra_callbacks.append(TemperatureCallback(rule_policy, rule_temperature_schedule))

    experience_data = QLearningDataset(maxlen=int(1e6))
    experience_data.update_data(decoder()) #need this as the DataLoader constructor won't accept an empty dataset
    experience_loader = DataLoader(dataset=experience_data,
                        batch_size=BATCH_SIZE, # we're dealing with single slices here, can afford this
                        shuffle=True,
                        collate_fn=collate_experiences)

    deepq_model = DeepQModelWrapper(decoder.stepper.model)

    fitter = get_fitter(deepq_model,
                        DeepQLoss(),  # last_reward_wgt=reward_sm),
                        experience_loader,
                            full_path(gen_save_file),
                            plot_prefix + 'deepq',
                            model_process_fun=None,#model_process_fun,
                            lr=lr_on,
                            extra_callbacks=gen_extra_callbacks)


    explore_metric_monitor = MetricPlotter(plot_prefix="",
                                   loss_display_cap=float('inf'),
                                   dashboard_name=dashboard,
                                   plot_ignore_initial=-1,
                                   process_model_fun=model_process_fun,
                                   smooth_weight=reward_sm)
    def explore():
        with torch.no_grad():
            runs = decoder()
        print('best reward in run:' + str(runs['rewards'].max().item()))
        experience_data.update_data(runs)
        explore_metric_monitor(None, decoder, runs, {}, torch.tensor(0.0))


    return explore, fitter
