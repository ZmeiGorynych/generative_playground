import numpy as np
import copy
import pandas as pd
import pickle, gzip
import datetime
from collections import OrderedDict
from generative_playground.utils.gpu_utils import get_gpu_memory_map
import os, inspect

class MetricPlotter:
    def __init__(self,
                 plot_prefix='',
                 save_file=None,
                 loss_display_cap=4,
                 dashboard_name=None,
                 plot_ignore_initial=0,
                 process_model_fun=None,
                 extra_metric_fun=None,
                 smooth_weight=0.0,
                 frequent_calls=False):
        self.plot_prefix = plot_prefix
        self.process_model_fun = process_model_fun
        self.extra_metric_fun = extra_metric_fun
        if save_file is not None:
            self.save_file = save_file
        else:
            self.save_file = plot_prefix.replace(' ','_') + '.pkz'

        my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        root_location = my_location + '/../'
        # TODO: pass that through in the constructor!
        self.save_file = root_location + 'molecules/train/vae/data/' + self.save_file
        self.plot_ignore_initial = plot_ignore_initial
        self.loss_display_cap = loss_display_cap
        self.plot_counter = 0
        self.stats = pd.DataFrame(columns=['batch', 'timestamp', 'gpu_usage', 'train', 'loss'])
        self.smooth_weight = smooth_weight
        self.smooth = {}
        try:
            from generative_playground.utils.visdom_helper import Dashboard
            if dashboard_name is not None:
                self.vis = Dashboard(dashboard_name,
                                     call_every=10 if frequent_calls else 1)
                self.have_visdom = True
        except:
            self.have_visdom = False
            self.vis = None

    def __call__(self,
                 inputs,
                 model,
                 outputs,
                 loss_fn,
                 loss):
                 # train,
                 # loss,
                 # metrics=None,
                 # model_out=None,
                 # inputs=None,
                 # targets=None):
        '''
        Plot the results of the latest batch
        :param train: bool: was this a traning batch?
        :param loss: float: latest loss
        :param metrics: dict {str:float} with any additional metrics
        :return: None
        '''
        print('calling metric monitor...')
        train = model.training
        loss = loss.data.item()
        metrics = loss_fn.metrics if hasattr(loss_fn, 'metrics') else None
        model_out = outputs

        if train:
            loss_name = self.plot_prefix + ' train_loss'
        else:
            loss_name = self.plot_prefix + ' val_loss'

        # show intermediate results
        gpu_usage = get_gpu_memory_map()
        print(loss_name, loss, self.plot_counter, gpu_usage)
        self.plot_counter += 1
        if self.vis is not None and self.plot_counter > self.plot_ignore_initial and self.have_visdom:
            all_metrics = {}
            # if not train: # don't want to call it too often as it takes time
            all_metrics['gpu_usage'] ={'type':'line',
                            'X': np.array([self.plot_counter]),
                            'Y':np.array([gpu_usage[0]])}
            all_metrics[loss_name] ={'type': 'line',
                            'X': np.array([self.plot_counter]),
                            'Y': np.array([min(self.loss_display_cap, loss)]),
                                'smooth':self.smooth_weight}
            if metrics is not None and len(metrics) > 0:
                all_metrics[loss_name + ' metrics']={'type':'line',
                            'X': np.array([self.plot_counter]),
                            'Y': np.array([[val for key, val in metrics.items()]]),
                            'opts':{'legend': [key for key, val in metrics.items()]},
                                    'smooth': self.smooth_weight}

            # if 'rewards' in outputs:
            #     rewards = outputs['rewards']
            #     if 'tensor' in str(type(rewards)):
            #         rewards = rewards.cpu().detach().numpy()
            #     all_metrics['reward'] = {'type': 'line',
            #                 'X': np.array([self.plot_counter]),
            #                 'Y': np.array([max(rewards)]),
            #                              'opts':{'legend':['max']},
            #                     'smooth':self.smooth_weight}

            if self.extra_metric_fun is not None:
                all_metrics.update(self.extra_metric_fun(inputs, targets, model_out, train, self.plot_counter))


            # now do the smooth:
            smoothed_metrics = {}
            for title, metric in all_metrics.items():
                if title not in self.smooth or 'smooth' not in metric:
                    self.smooth[title] = metric
                else:
                    self.smooth[title] = smooth_data(self.smooth[title], metric, metric['smooth'])

            self.vis.plot_metric_dict({title: value for title, value in self.smooth.items() if title in all_metrics.keys()})

            # TODO: factor this out
            if self.process_model_fun is not None:
                self.process_model_fun(model_out, self.vis, self.plot_counter)



        metrics =  {} if metrics is None else copy.copy(metrics)
        metrics['train'] = train
        metrics['gpu_usage'] = gpu_usage[0]
        metrics['loss'] = loss
        metrics['batch'] = self.plot_counter
        metrics['timestamp'] = datetime.datetime.now()

        self.stats = self.stats.append(metrics, ignore_index=True)

        if True:#not train: # only save to disk during valdation calls for speedup
            with gzip.open(self.save_file,'wb') as f:
                pickle.dump(self.stats, f)


def smooth_data(smoothed, metric, w):
    if 'opts' in metric: # need to match the legend entries, they may vary by batch
        sm_data = OrderedDict(zip(smoothed['opts']['legend'], smoothed['Y'].T))
        new_data = OrderedDict(zip(metric['opts']['legend'], metric['Y'].T))
        # compose all sm legend entries
        for key,value in new_data.items():
            if key not in sm_data:
                sm_data[key] = value
            elif np.isnan(value):
                pass
            else:
                sm_data[key] = w*sm_data[key] + (1-w)*value
        sm_legend =[]
        sm_Y = []
        for key, value in sm_data.items():
            sm_legend.append(key)
            sm_Y.append(value)

        smoothed['opts']['legend'] = sm_legend
        smoothed['Y'] = np.array(sm_Y).T
    else:
        smoothed['Y'] = w*smoothed['Y'] + (1-w)*metric['Y']
    smoothed['X'] = metric["X"]
    return smoothed