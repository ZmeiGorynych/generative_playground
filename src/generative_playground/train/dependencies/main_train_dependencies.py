import os, inspect
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import copy
import pickle

from transformer.Models import Encoder
from generative_playground.utils.fit import fit
from generative_playground.models.losses.multiple_cross_entropy_loss import MultipleCrossEntropyLoss
from generative_playground.utils.gpu_utils import use_gpu, to_gpu
from generative_playground.utils.metric_monitor import MetricPlotter
from generative_playground.utils.checkpointer import Checkpointer
from generative_playground.data_utils.data_sources import MultiDatasetFromHDF5, train_valid_loaders, IterableTransform
from generative_playground.models.heads.multiple_output_head import MultipleOutputHead

def train_dependencies(EPOCHS=None,
                          BATCH_SIZE=None,
                          max_steps=None,
                          feature_len = None,
                          lr=2e-4,
                          drop_rate = 0.0,
                          plot_ignore_initial = 0,
                          save_file = None,
                          preload_file = None,
                        meta=None,
                          decoder_type='action',
                          plot_prefix = '',
                          dashboard = 'policy gradient'):

    root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_location = root_location + '/../'
    save_path = root_location + 'pretrained/' + save_file

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

    pre_model = Encoder(len(meta['emb_index']),
            max_steps,
            dropout=drop_rate,
            padding_idx=0)

    model = MultipleOutputHead(pre_model,
                               [len(meta['emb_index']),# word
                                meta['maxlen'],# head
                                len(meta['upos']),# part of speech
                                len(meta['deprel']) # dependency relationship
                                ],
                               drop_rate=drop_rate,
                               labels=['token', 'head', 'upos', 'deprel'])

    model = to_gpu(model)

    if preload_file is not None:
        try:
            preload_path = root_location + 'pretrained/' + preload_file
            model.load_state_dict(torch.load(preload_path))
        except:
            pass

    def model_process_fun(model_out, visdom, n):
        pass
        # if mol is not None:
        #
        #     scores, norm_scores = scorer.get_scores([this_smile])
        #     visdom.append('score component',
        #                     'line',
        #                     X=np.array([n]),
        #                     Y=np.array([[x for x in norm_scores[0]] + [norm_scores[0].sum()] + [scores[0].sum()] + [desc.CalcNumAromaticRings(mol)]]),
        #                     opts={'legend': ['logP','SA','cycle','norm_reward','reward','Aromatic rings']})
        #     visdom.append('fraction valid',
        #                   'line',
        #                   X=np.array([n]),
        #                   Y=np.array([valid.mean().data.item()]))

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
                                    save_always=False)

        fitter = fit(train_gen=train_gen,
                    valid_gen=valid_gen,
                    model = model,
                    optimizer = optimizer,
                    scheduler = scheduler,
                    epochs = EPOCHS,
                    loss_fn = loss_obj,
                    batches_to_valid=9,
                    metric_monitor = metric_monitor,
                    checkpointer = checkpointer
                        )

        return fitter

    # TODO: need to be cleaner about dataset creation
    with open('../../ud_utils/data.pickle', 'rb') as f:
        # a simple array implements the __len__ and __getitem__ methods, can we just use that?
        main_dataset = pickle.load(f)

    train_loader, valid_loader = train_valid_loaders(main_dataset,
                                                     valid_fraction=0.1,
                                                     batch_size=BATCH_SIZE,
                                                     pin_memory=use_gpu)

    target_fields = ['head', 'upos', 'deprel']

    def nice_loader(loader):
        return IterableTransform(loader,
                                 lambda x: (to_gpu(x['token']),
                                            {key:to_gpu(val) for key, val in x.items() if key in target_fields}))

    # the on-policy fitter
    fitter1 = get_fitter(model,
                         nice_loader(train_loader),
                         nice_loader(valid_loader),
                         MultipleCrossEntropyLoss(),
                         plot_prefix + 'on-policy',
                         model_process_fun=model_process_fun,
                         lr=lr)


    return model, fitter1

