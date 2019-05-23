import os, inspect
from collections import deque
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from torch.utils.data import DataLoader
from generative_playground.molecules.molecule_saver_callback import MoleculeSaver

from generative_playground.utils.fit_rl import fit_rl
from generative_playground.data_utils.data_sources import DatasetFromHDF5
from generative_playground.utils.gpu_utils import use_gpu, to_gpu
from generative_playground.molecules.model_settings import get_settings
from generative_playground.metrics.metric_monitor import MetricPlotter
from generative_playground.utils.checkpointer import Checkpointer
from generative_playground.models.problem.rl.task import SequenceGenerationTask
from generative_playground.models.decoder.decoder import get_decoder
from generative_playground.models.losses.policy_gradient_loss import PolicyGradientLoss
from generative_playground.models.decoder.policy import SoftmaxRandomSamplePolicy, PolicyFromTarget
from generative_playground.data_utils.blended_dataset import EvenlyBlendedDataset
from generative_playground.data_utils.data_sources import IncrementingHDF5Dataset
from generative_playground.codec.codec import get_codec
from generative_playground.molecules.data_utils.zinc_utils import get_smiles_from_database
from generative_playground.data_utils.data_sources import IterableTransform
from generative_playground.molecules.models.graph_discriminator import GraphDiscriminator
from generative_playground.utils.gpu_utils import device


def train_policy_gradient(molecules=True,
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
                          reward_sm = 0.0,
                          preload_file_root_name=None,
                          anchor_file=None,
                          anchor_weight=0.0,
                          decoder_type='action',
                          plot_prefix='',
                          dashboard='policy gradient',
                          smiles_save_file=None,
                          on_policy_loss_type='best',
                          priors=True,
                          node_temperature_schedule=lambda x: 1.0,
                          rule_temperature_schedule=lambda x: 1.0,
                          eps=0.0,
                          half_float=False,
                          extra_repetition_penalty=0.0):
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
    codec = get_codec(molecules, grammar, settings['max_seq_length'])
    discrim_model = GraphDiscriminator(codec.grammar, drop_rate=drop_rate)
    if False and preload_file_root_name is not None:
        try:
            preload_path = full_path(disc_preload_file)
            discrim_model.load_state_dict(torch.load(preload_path), strict=False)
            print('Discriminator weights loaded successfully!')
        except Exception as e:
            print('failed to load discriminator weights ' + str(e))


    zinc_data = get_smiles_from_database(source=smiles_source)
    zinc_set = set(zinc_data)
    lookbacks = [BATCH_SIZE, 10*BATCH_SIZE, 100*BATCH_SIZE]
    history_data = [deque(['O'], maxlen=lb) for lb in lookbacks]

    def originality_mult(smiles_list):
        out = []
        for s in smiles_list:
            if s in zinc_set:
                out.append(0.5)
            elif s in history_data[0]:
                out.append(0.5)
            elif s in history_data[1]:
                out.append(0.70)
            elif s in history_data[2]:
                out.append(0.85)
            else:
                out.append(1.0)
        return np.array(out)


    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def discriminator_reward_mult(smiles_list):
        orig_state = discrim_model.training
        discrim_model.eval()
        discrim_out_logits = discrim_model(smiles_list)['p_zinc']
        discrim_probs = F.softmax(discrim_out_logits, dim=1)
        prob_zinc = discrim_probs[:,1].detach().cpu().numpy()
        if orig_state:
            discrim_model.train()
        return prob_zinc


    def apply_originality_penalty(x, orig_mult):
        assert x <= 1, "Reward must be no greater than 0"
        if x > 0.5: # want to punish nearly-perfect scores less and less
            out = math.pow(x, 1/orig_mult)
        else: # continuous join at 0.5
            penalty = math.pow(0.5, 1/orig_mult) - 0.5
            out = x + penalty

        out -= extra_repetition_penalty*(1-1/orig_mult)
        return out


    def adj_reward(x):
        if discrim_wt > 1e-5:
            p = discriminator_reward_mult(x)
        else:
            p = 0
        rwd = np.array(reward_fun_on(x))
        orig_mult = originality_mult(x)
        # we assume the reward is <=1, first term will dominate for reward <0, second for 0 < reward < 1
        # reward = np.minimum(rwd/orig_mult, np.power(np.abs(rwd),1/orig_mult))
        reward = np.array([apply_originality_penalty(x, om) for x, om in zip(rwd, orig_mult)])
        out = reward + discrim_wt*p*orig_mult
        return out

    def adj_reward_old(x):
        p = discriminator_reward_mult(x)
        w = sigmoid(-(p-p_thresh)/0.01)
        if randomize_reward:
            rand = np.random.uniform(size=p.shape)
            w *= rand
        reward = np.maximum(reward_fun_on(x), p_thresh)
        weighted_reward = w * p + (1 - w) * reward
        out = weighted_reward * originality_mult(x) #
        return out

    if EPOCHS is not None:
        settings['EPOCHS'] = EPOCHS
    if BATCH_SIZE is not None:
        settings['BATCH_SIZE'] = BATCH_SIZE

    task = SequenceGenerationTask(molecules=molecules,
                                  grammar=grammar,
                                  reward_fun=adj_reward,
                                  batch_size=BATCH_SIZE,
                                  max_steps=max_steps,
                                  save_dataset=None)

    node_policy = SoftmaxRandomSamplePolicy(temperature=torch.tensor(1.0), eps=eps)
    rule_policy = SoftmaxRandomSamplePolicy(temperature=torch.tensor(1.0), eps=eps)

    model = get_decoder(molecules,
                        grammar,
                        z_size=settings['z_size'],
                        decoder_hidden_n=200,
                        feature_len=codec.feature_len(),
                        max_seq_length=max_steps,
                        drop_rate=drop_rate,
                        batch_size=BATCH_SIZE,
                        decoder_type=decoder_type,
                        reward_fun=adj_reward,
                        task=task,
                        node_policy=node_policy,
                        rule_policy=rule_policy,
                        priors=priors)[0]

    if preload_file_root_name is not None:
        try:
            preload_path = full_path(gen_preload_file)
            model.load_state_dict(torch.load(preload_path, map_location='cpu'), strict=False)
            print('Generator weights loaded successfully!')
        except Exception as e:
            print('failed to load generator weights ' + str(e))

    anchor_model = None

    from generative_playground.molecules.rdkit_utils.rdkit_utils import NormalizedScorer
    import rdkit.Chem.rdMolDescriptors as desc
    import numpy as np
    scorer = NormalizedScorer()

    def model_process_fun(model_out, visdom, n):
        # TODO: rephrase this to return a dict, instead of calling visdom directly
        from rdkit import Chem
        from rdkit.Chem.Draw import MolToFile
        # actions, logits, rewards, terminals, info = model_out
        smiles, valid = model_out['info']
        total_rewards = model_out['rewards']
        if len(total_rewards.shape) > 1:
            total_rewards = total_rewards.sum(1)
        best_ind = torch.argmax(total_rewards).data.item()
        this_smile = smiles[best_ind]
        mol = Chem.MolFromSmiles(this_smile)
        pic_save_path = root_location + 'images/' + 'tmp.svg'
        if mol is not None:
            try:
                MolToFile(mol, pic_save_path, imageType='svg')
                with open(pic_save_path, 'r') as myfile:
                    data = myfile.read()
                data = data.replace('svg:', '')
                visdom.append('best molecule of batch', 'svg', svgstr=data)
            except Exception as e:
                print(e)
            scores, norm_scores = scorer.get_scores([this_smile])
            visdom.append('score component',
                          'line',
                          X=np.array([n]),
                          Y=np.array([ [x for x in norm_scores[0]] + [norm_scores[0].sum()] + [scores[0].sum()] + [
                              desc.CalcNumAromaticRings(mol)] + [total_rewards[best_ind].item()]]),
                          opts={'legend': ['logP', 'SA', 'cycle', 'norm_reward', 'reward', 'Aromatic rings','eff_reward']})
            visdom.append('fraction valid',
                          'line',
                          X=np.array([n]),
                          Y=np.array([valid.mean().data.item()]))
            visdom.append('num atoms', 'line',
                          X=np.array([n]),
                          Y=np.array([len(mol.GetAtoms())]))

    if reward_fun_off is None:
        reward_fun_off = reward_fun_on

    # construct the loader to feed the discriminator
    def make_callback(data):
        def hc(inputs, model, outputs, loss_fn, loss):
            graphs = outputs['graphs']
            smiles = [g.to_smiles() for g in graphs]
            for s in smiles: # only store unique instances of molecules so discriminator can't guess on frequency
                if s not in data:
                    data.append(s)
        return hc

    class TemperatureCallback:
        def __init__(self, policy, temperature_function):
            self.policy = policy
            self.counter = 0
            self.temp_fun = temperature_function

        def __call__(self,inputs, model, outputs, loss_fn, loss):
            self.counter +=1
            target_temp = self.temp_fun(self.counter)
            self.policy.set_temperature(target_temp)


    # need to have something there to begin with, else the DataLoader constructor barfs

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

    class GeneratorToIterable:
        def __init__(self, gen):
            self.gen = gen
            # we assume the generator is finite
            self.len = 0
            for _ in gen():
                self.len+=1

        def __len__(self):
            return self.len

        def __iter__(self):
            return self.gen()


    def my_gen():
        for _ in range(1000):
            yield to_gpu(torch.zeros(BATCH_SIZE, settings['z_size']))

    # the on-policy fitter

    gen_extra_callbacks = [make_callback(d) for d in history_data]

    if smiles_save_file is not None:
        smiles_save_path = os.path.realpath(root_location + 'pretrained/' + smiles_save_file)
        gen_extra_callbacks.append(MoleculeSaver(smiles_save_path, gzip=True))
        print('Saved SMILES to {}'.format(smiles_save_file))

    if node_temperature_schedule is not None:
        gen_extra_callbacks.append(TemperatureCallback(node_policy, node_temperature_schedule))

    if rule_temperature_schedule is not None:
        gen_extra_callbacks.append(TemperatureCallback(rule_policy, rule_temperature_schedule))

    fitter1 = get_rl_fitter(model,
                            PolicyGradientLoss(on_policy_loss_type),# last_reward_wgt=reward_sm),
                            GeneratorToIterable(my_gen),
                            full_path(gen_save_file),
                            plot_prefix + 'on-policy',
                            model_process_fun=model_process_fun,
                            lr=lr_on,
                            extra_callbacks=gen_extra_callbacks,
                            anchor_model=anchor_model,
                            anchor_weight=anchor_weight)
    #
    # # get existing molecule data to add training
    pre_dataset = EvenlyBlendedDataset(2 * [history_data[0]] + history_data[1:] , labels=False)  # a blend of 3 time horizons
    dataset = EvenlyBlendedDataset([pre_dataset,zinc_data], labels=True)
    discrim_loader = DataLoader(dataset, shuffle=True, batch_size=50)

    class MyLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.celoss = nn.CrossEntropyLoss()

        def forward(self, x):
            # tmp = discriminator_reward_mult(x['smiles'])
            # tmp2 = F.softmax(x['p_zinc'], dim=1)[:,1].detach().cpu().numpy()
            # import numpy as np
            # assert np.max(np.abs(tmp-tmp2)) < 1e-6
            return self.celoss(x['p_zinc'].to(device), x['dataset_index'].to(device))

    fitter2 = get_rl_fitter(discrim_model,
                            MyLoss(),
                            IterableTransform(discrim_loader,
                                              lambda x: {'smiles': x['X'], 'dataset_index': x['dataset_index']}),
                            full_path(disc_save_file),
                            plot_prefix + ' discriminator',
                            lr=lr_discrim,
                            model_process_fun=None)

    def on_policy_gen(fitter, model):
        while True:
            # model.policy = SoftmaxRandomSamplePolicy()#bias=codec.grammar.get_log_frequencies())
            yield next(fitter)

    return model, fitter1, fitter2  #,on_policy_gen(fitter1, model)
