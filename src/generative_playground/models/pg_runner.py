import os, inspect
from collections import deque
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import gzip, dill, cloudpickle
import copy

from generative_playground.models.reward_adjuster import adj_reward, AdjustedRewardCalculator
from generative_playground.models.temperature_schedule import TemperatureCallback
from generative_playground.molecules.molecule_saver_callback import MoleculeSaver
from generative_playground.molecules.visualize_molecules import model_process_fun

from generative_playground.utils.fit_rl import fit_rl
from generative_playground.utils.gpu_utils import to_gpu
from generative_playground.molecules.model_settings import get_settings
from generative_playground.metrics.metric_monitor import MetricPlotter
from generative_playground.models.problem.rl.task import SequenceGenerationTask
from generative_playground.models.decoder.decoder import get_decoder
from generative_playground.models.losses.policy_gradient_loss import PolicyGradientLoss
from generative_playground.models.problem.policy import SoftmaxRandomSamplePolicy, SoftmaxRandomSamplePolicySparse
from generative_playground.codec.codec import get_codec
from generative_playground.molecules.data_utils.zinc_utils import get_smiles_from_database
from generative_playground.data_utils.data_sources import GeneratorToIterable

class Saveable:
    def save(self):
        print('saving to ' + self.save_file_name + '...')
        with gzip.open(self.save_file_name, 'wb') as f:
            dill.dump(self, f)
        print('done!')
        return self.save_file_name


    @classmethod
    def load(cls, save_file_name):
        print('loading from ' + save_file_name + '...')
        with gzip.open(save_file_name, 'rb') as f:
            inst = dill.load(f)
        print('done!')
        return inst


class PolicyGradientRunner(Saveable):
    def __init__(self,
                 grammar,
                 smiles_source='ZINC',
                 BATCH_SIZE=None,
                 reward_fun=None,
                 max_steps=277,
                 num_batches=100,
                 lr=2e-4,
                 entropy_wgt=1.0,
                 lr_schedule=None,
                 root_name=None,
                 preload_file_root_name=None,
                 save_location=None,
                 plot_metrics=True,
                 metric_smooth=0.0,
                 decoder_type='graph_conditional',
                 on_policy_loss_type='advantage_record',
                 priors='conditional',
                 rule_temperature_schedule=None,
                 eps=0.0,
                 half_float=False,
                 extra_repetition_penalty=0.0):

        self.num_batches = num_batches
        self.save_location = save_location
        self.molecule_saver = MoleculeSaver(None, gzip=True)
        self.metric_monitor = None # to be populated by self.set_root_name(...)

        zinc_data = get_smiles_from_database(source=smiles_source)
        zinc_set = set(zinc_data)
        lookbacks = [BATCH_SIZE, 10 * BATCH_SIZE, 100 * BATCH_SIZE]
        history_data = [deque(['O'], maxlen=lb) for lb in lookbacks]

        if root_name is not None:
            pass
            # gen_save_file = root_name + '_gen.h5'
        if preload_file_root_name is not None:
            gen_preload_file = preload_file_root_name + '_gen.h5'

        settings = get_settings(molecules=True, grammar=grammar)
        codec = get_codec(True, grammar, settings['max_seq_length'])

        if BATCH_SIZE is not None:
            settings['BATCH_SIZE'] = BATCH_SIZE

        self.alt_reward_calc = AdjustedRewardCalculator(reward_fun, zinc_set, lookbacks, extra_repetition_penalty, 0,
                                                   discrim_model=None)
        self.reward_fun = lambda x: adj_reward(0,
                                               None,
                                               reward_fun,
                                               zinc_set,
                                               history_data,
                                               extra_repetition_penalty,
                                               x,
                                               alt_calc=self.alt_reward_calc)

        task = SequenceGenerationTask(molecules=True,
                                      grammar=grammar,
                                      reward_fun=self.alt_reward_calc,
                                      batch_size=BATCH_SIZE,
                                      max_steps=max_steps,
                                      save_dataset=None)

        if 'sparse' in decoder_type:
            rule_policy = SoftmaxRandomSamplePolicySparse()
        else:
            rule_policy = SoftmaxRandomSamplePolicy(temperature=torch.tensor(1.0), eps=eps)

        # TODO: strip this down to the normal call
        self.model = get_decoder(True,
                            grammar,
                            z_size=settings['z_size'],
                            decoder_hidden_n=200,
                            feature_len=codec.feature_len(),
                            max_seq_length=max_steps,
                            batch_size=BATCH_SIZE,
                            decoder_type=decoder_type,
                            reward_fun=self.alt_reward_calc,
                            task=task,
                            rule_policy=rule_policy,
                            priors=priors)[0]

        if preload_file_root_name is not None:
            try:
                preload_path = os.path.realpath(save_location + gen_preload_file)
                self.model.load_state_dict(torch.load(preload_path, map_location='cpu'), strict=False)
                print('Generator weights loaded successfully!')
            except Exception as e:
                print('failed to load generator weights ' + str(e))

        # construct the loader to feed the discriminator
        def make_callback(data):
            def hc(inputs, model, outputs, loss_fn, loss):
                graphs = outputs['graphs']
                smiles = [g.to_smiles() for g in graphs]
                for s in smiles: # only store unique instances of molecules so discriminator can't guess on frequency
                    if s not in data:
                        data.append(s)
            return hc

        if plot_metrics:
            # TODO: save_file for rewards data goes here?
            self.metric_monitor_factory = lambda name: MetricPlotter(plot_prefix='',
                                           loss_display_cap=float('inf'),
                                           dashboard_name=name,
                                                                     save_location=save_location,
                                           process_model_fun=model_process_fun,
                                           smooth_weight=metric_smooth)
        else:
            self.metric_monitor_factory = lambda x: None


        # the on-policy fitter

        gen_extra_callbacks = [make_callback(d) for d in history_data]
        gen_extra_callbacks.append(self.molecule_saver)
        if rule_temperature_schedule is not None:
            gen_extra_callbacks.append(TemperatureCallback(rule_policy, rule_temperature_schedule))

        nice_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(nice_params, lr=lr, eps=1e-4)

        if lr_schedule is None:
            lr_schedule = lambda x: 1.0
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_schedule)
        self.loss = PolicyGradientLoss(on_policy_loss_type, entropy_wgt=entropy_wgt)
        self.fitter_factory = lambda: make_fitter(BATCH_SIZE, settings['z_size'], [self.metric_monitor] + gen_extra_callbacks, self)

        self.fitter = self.fitter_factory()
        self.set_root_name(root_name)
        print('Runner initialized!')



    def run(self):
        for i in range(self.num_batches):
            next(self.fitter)
        out = self.save()
        return out

    def set_root_name(self, root_name):
        self.root_name = root_name

        smiles_save_file = root_name + '_smiles.zip'
        smiles_save_path = os.path.realpath(self.save_location + '/' + smiles_save_file)
        self.molecule_saver.filename = smiles_save_path
        print('Saving SMILES to {}'.format(smiles_save_path))

        self.fitter.gi_frame.f_locals['callbacks'][0] = self.metric_monitor_factory(root_name)
        print('publishing to ' + root_name)

        self.save_file_name = os.path.realpath(self.save_location + '/' + root_name + '_runner.zip')
        print('Runner to be saved to ' + self.save_file_name)


    def __getstate__(self):
        state = {key: value for key, value in self.__dict__.items() if key != 'fitter'}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # need to use the factory because neither dill nor cloudpickle will serialize generators
        self.fitter = self.fitter_factory()

    def get_model_coeff_vector(self):
        coeffvec = self.model.stepper.model.get_params_as_vector()
        return coeffvec

    def set_model_coeff_vector(self, vector_in):
        self.model.stepper.model.set_params_from_vector(vector_in)

    @property
    def params(self):
        return self.get_model_coeff_vector()

    @params.setter
    def params(self, vector_in):
        self.set_model_coeff_vector(vector_in)

    @classmethod
    def load_from_root_name(cls, save_location, root_name):
        full_save_file = os.path.realpath(save_location + '/' + root_name + '_runner.zip')
        inst = cls.load(full_save_file)
        return inst

def make_fitter(batch_size, z_size, callbacks, obj):
    def my_gen(length=100):
        for _ in range(length):
            yield to_gpu(torch.zeros(batch_size, z_size)) #settings['z_size']

    fitter = fit_rl(train_gen=GeneratorToIterable(my_gen),
                         model=obj.model,
                         optimizer=obj.optimizer,
                         scheduler=obj.scheduler,
                         loss_fn=obj.loss,
                         grad_clip=5,
                         callbacks=callbacks
                         )
    return fitter