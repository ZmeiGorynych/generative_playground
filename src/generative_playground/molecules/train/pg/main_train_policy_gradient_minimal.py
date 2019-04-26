import os, inspect
from collections import deque
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

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
from generative_playground.molecules.data_utils.zinc_utils import get_zinc_smiles
from generative_playground.data_utils.data_sources import IterableTransform
from generative_playground.molecules.models.graph_discriminator import GraphDiscriminator
from generative_playground.utils.gpu_utils import device

def train_policy_gradient(molecules=True,
                          grammar=True,
                          EPOCHS=None,
                          BATCH_SIZE=None,
                          reward_fun_on=None,
                          reward_fun_off=None,
                          max_steps=277,
                          lr_on=2e-4,
                          lr_off=1e-4,
                          drop_rate=0.0,
                          plot_ignore_initial=0,
                          save_file=None,
                          preload_file=None,
                          anchor_file=None,
                          anchor_weight=0.0,
                          decoder_type='action',
                          plot_prefix='',
                          dashboard='policy gradient',
                          smiles_save_file=None,
                          on_policy_loss_type='best',
                          off_policy_loss_type='mean',
                          sanity_checks=True):
    root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_location = root_location + '/../'
    save_path = root_location + 'pretrained/' + save_file
    smiles_save_path = root_location + 'pretrained/' + smiles_save_file

    settings = get_settings(molecules=molecules, grammar=grammar)
    codec = get_codec(molecules, grammar, settings['max_seq_length'])

    if EPOCHS is not None:
        settings['EPOCHS'] = EPOCHS
    if BATCH_SIZE is not None:
        settings['BATCH_SIZE'] = BATCH_SIZE

    save_dataset = IncrementingHDF5Dataset(smiles_save_path)

    task = SequenceGenerationTask(molecules=molecules,
                                  grammar=grammar,
                                  reward_fun=reward_fun_on,
                                  batch_size=BATCH_SIZE,
                                  max_steps=max_steps,
                                  save_dataset=save_dataset)

    model = get_decoder(molecules,
                        grammar,
                        z_size=settings['z_size'],
                        decoder_hidden_n=200,
                        feature_len=codec.feature_len(),
                        max_seq_length=max_steps,
                        drop_rate=drop_rate,
                        decoder_type=decoder_type,
                        task=task)[0]
    # if preload_file is not None:
    #     try:
    #         preload_path = root_location + 'pretrained/' + preload_file
    #         model.load_state_dict(torch.load(preload_path))
    #     except:
    #         pass

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
        total_rewards = model_out['rewards'].sum(1)
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
                          Y=np.array([[x for x in norm_scores[0]] + [norm_scores[0].sum()] + [scores[0].sum()] + [
                              desc.CalcNumAromaticRings(mol)]]),
                          opts={'legend': ['logP', 'SA', 'cycle', 'norm_reward', 'reward', 'Aromatic rings']})
            visdom.append('fraction valid',
                          'line',
                          X=np.array([n]),
                          Y=np.array([valid.mean().data.item()]))

    if reward_fun_off is None:
        reward_fun_off = reward_fun_on

    # construct the loader to feed the discriminator
    history_size = 1000
    history_data = deque(['O'],
                         maxlen=history_size)  # need to have something there to begin with, else the DataLoader constructor barfs

    def history_callback(inputs, model, outputs, loss_fn, loss):
        graphs = outputs['graphs']
        smiles = [g.to_smiles() for g in graphs]
        history_data.extend(smiles)



    def get_rl_fitter(model,
                      loss_obj,
                      train_gen,
                      fit_plot_prefix='',
                      model_process_fun=None,
                      lr=None,
                      extra_callbacks=[],
                      loss_display_cap=float('inf'),
                      anchor_model=None,
                      anchor_weight=0
                      ):
        nice_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(nice_params, lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)

        if dashboard is not None:
            metric_monitor = MetricPlotter(plot_prefix=fit_plot_prefix,
                                           loss_display_cap=loss_display_cap,
                                           dashboard_name=dashboard,
                                           plot_ignore_initial=plot_ignore_initial,
                                           process_model_fun=model_process_fun)
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

    #
    fitter1 = get_rl_fitter(model,
                            PolicyGradientLoss(on_policy_loss_type),
                            GeneratorToIterable(my_gen),
                            plot_prefix + 'on-policy',
                            model_process_fun=model_process_fun,
                            lr=lr_on,
                            extra_callbacks=[history_callback],
                            anchor_model=anchor_model,
                            anchor_weight=anchor_weight)
    #
    # # get existing molecule data to add training
    zinc_data = get_zinc_smiles()
    dataset = EvenlyBlendedDataset([history_data, zinc_data], labels=True)
    discrim_loader = DataLoader(dataset, shuffle=True, batch_size=10)
    discrim_model = GraphDiscriminator(model.stepper.mask_gen.grammar, drop_rate=drop_rate)
    celoss = nn.CrossEntropyLoss()
    loss = lambda x: celoss(x['p_zinc'].to(device), x['dataset_index'].to(device))
    fitter2 = get_rl_fitter(discrim_model,
                            loss,
                            IterableTransform(discrim_loader,
                                              lambda x: {'smiles': x['X'], 'dataset_index': x['dataset_index']}),
                            plot_prefix + ' discriminator',
                            lr=lr_off,
                            model_process_fun=None)

    def on_policy_gen(fitter, model):
        while True:
            model.policy = SoftmaxRandomSamplePolicy(bias=codec.grammar.get_log_frequencies())
            yield next(fitter)

    return model, on_policy_gen(fitter1, model), fitter2