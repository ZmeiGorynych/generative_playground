import numpy as np
import copy
import pandas as pd
import pickle, gzip
import datetime
from collections import OrderedDict
from generative_playground.utils.gpu_utils import get_gpu_memory_map, get_free_ram
import os, inspect


class MetricStats:
    def __init__(self):
        self.best_reward = float('-inf')

    def __call__(self, rewards, smiles=None):
        """
        Calculates reward stats
        :param rewards: numpy array of rewards
        :param smiles: array of strings
        :return: dict with the metrics
        """
        self.best_reward = max(self.best_reward, max(rewards))
        metrics = {'rec rwd': self.best_reward,
                   'avg rwd': rewards.mean(),
                   'max rwd': rewards.max(),
                   'med rwd': np.median(rewards)
                   }
        if smiles is not None: metrics['unique'] = len(set(smiles)) / len(smiles)

        return metrics

class MetricPlotter:
    def __init__(self,
                 plot_prefix='',
                 save_file=None,
                 loss_display_cap=4,
                 dashboard_name='main',
                 save_location=None,
                 plot_ignore_initial=0,
                 process_model_fun=None,
                 extra_metric_fun=None,
                 smooth_weight=0.0,
                 frequent_calls=False):
        self.plot_prefix = plot_prefix
        self.process_model_fun = process_model_fun
        self.extra_metric_fun = extra_metric_fun


        if save_location is not None:
            if save_file is None:
                save_file = '/' + dashboard_name + '_metrics.zip'
            self.save_file = save_location + save_file
        else:
            self.save_file = None
        self.plot_ignore_initial = plot_ignore_initial
        self.loss_display_cap = loss_display_cap
        self.plot_counter = 0
        self.reward_calc = MetricStats()
        self.stats = pd.DataFrame(columns=['batch', 'timestamp', 'gpu_usage', 'train', 'loss'])
        self.smooth_weight = smooth_weight
        self.smooth = {}
        self.last_timestamp = datetime.datetime.now()
        self.dashboard_name = dashboard_name
        self.frequent_calls = frequent_calls
        self.vis = None
        self.have_visdom = False
        self.init_visdom()

    def init_visdom(self):
        try:
            from generative_playground.utils.visdom_helper import Dashboard
            if self.dashboard_name is not None:
                self.vis = Dashboard(self.dashboard_name,
                                     call_every=10 if self.frequent_calls else 1)
                self.have_visdom = True
            else:
                self.vis = None
                self.have_visdom = False
        except:
            self.have_visdom = False
            self.vis = None

    def __getstate__(self):
        state = {key:value for key, value in self.__dict__.items() if key != 'vis'}
        state['plots'] = self.vis.plots
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.init_visdom()
        self.vis.plots = state['plots']
        self.plot_counter += 1

    def __call__(self,
                 _, #inputs
                 model,
                 outputs,
                 loss_fn,
                 loss):
        '''
        Plot the results of the latest batch
        :param train: bool: was this a traning batch?
        :param loss: float: latest loss
        :param metrics: dict {str:float} with any additional metrics
        :return: None
        '''
        print('calling metric monitor...')
        try:
            train = model.training
        except:
            train = True

        if hasattr(loss, 'device'):
            loss = loss.data.item()

        metrics_from_loss = loss_fn.metrics if hasattr(loss_fn, 'metrics') else None
        model_out = outputs

        if train:
            loss_name = self.plot_prefix + ' train_loss'
        else:
            loss_name = self.plot_prefix + ' val_loss'

        # show intermediate results
        gpu_usage = get_gpu_memory_map()
        cpu_usage = get_free_ram()
        print(loss_name, loss, self.plot_counter, gpu_usage)
        self.plot_counter += 1
        if self.vis is not None and self.plot_counter > self.plot_ignore_initial and self.have_visdom:
            all_metrics = {}
            # if not train: # don't want to call it too often as it takes time
            all_metrics['memory_usage'] = self.entry_from_dict({'gpu': gpu_usage[0],'cpu': cpu_usage['used']})
                             #   {'type':'line',
                            # 'X': np.array([self.plot_counter]),
                            # 'Y':np.array([gpu_usage[0]])}
            if loss is not None:
                all_metrics[loss_name] ={'type': 'line',
                            'X': np.array([self.plot_counter]),
                            'Y': np.array([min(self.loss_display_cap, loss)]),
                                'smooth':self.smooth_weight}

            if metrics_from_loss is not None and len(metrics_from_loss) > 0:
                for key, value in metrics_from_loss.items():
                    if type(value) == dict: # dict of dicts, so multiple plots
                        all_metrics[key] = self.entry_from_dict(value)
                    else: # just one dict with data, old-style
                        all_metrics[loss_name + ' metrics'] = self.entry_from_dict(metrics_from_loss)
                        break
            now = datetime.datetime.now()
            if self.last_timestamp is not None:
                batch_duration = (now - self.last_timestamp).total_seconds()
                all_metrics['seconds_per_batch'] = {'type':'line',
                            'X': np.array([self.plot_counter]),
                            'Y':np.array([batch_duration])}
            self.last_timestamp = now

            try:
                smiles = outputs['info'][0]
            except:
                smiles = None

            try:
                rewards = outputs['rewards']
                if len(rewards.shape) == 2:
                    rewards = rewards.sum(1)
                rewards = to_numpy(rewards)
                reward_dict = self.reward_calc(rewards,smiles)
                all_metrics['reward_stats'] = self.entry_from_dict(reward_dict)
            except:
                rewards = np.array([0])
            # if self.extra_metric_fun is not None:
            #     all_metrics.update(self.extra_metric_fun(inputs, targets, model_out, train, self.plot_counter))


            # now do the smooth:
            smoothed_metrics = {}
            for title, metric in all_metrics.items():
                if 'smooth' not in metric:
                    self.smooth[title] = metric
                else:
                    self.smooth[title] = smooth_data(self.smooth[title] if title in self.smooth else metric,
                                                     metric,
                                                     metric['smooth'])

            self.vis.plot_metric_dict({title: value for title, value in self.smooth.items() if title in all_metrics.keys()})

            # TODO: factor this out
            if self.process_model_fun is not None:
                self.process_model_fun(model_out, self.vis, self.plot_counter)



        metrics_from_loss =  {} if metrics_from_loss is None else copy.copy(metrics_from_loss)
        metrics_from_loss['train'] = train
        metrics_from_loss['gpu_usage'] = gpu_usage[0]
        metrics_from_loss['loss'] = loss
        metrics_from_loss['batch'] = self.plot_counter
        metrics_from_loss['timestamp'] = datetime.datetime.now()
        metrics_from_loss['best_reward'] = rewards.max()

        self.stats = self.stats.append(metrics_from_loss, ignore_index=True)

        if self.save_file is not None:
            with gzip.open(self.save_file,'wb') as f:
                pickle.dump(self.stats, f)

    def entry_from_dict(self, metrics):
        return {'type': 'line',
         'X': np.array([self.plot_counter]),
         'Y': np.array([[val for key, val in metrics.items()]]),
         'opts': {'legend': [key for key, val in metrics.items()]},
         'smooth': self.smooth_weight}

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

def to_numpy(x):
    if hasattr(x,'device'): # must be pytorch
        return x.cpu().detach().numpy()
    else:
        return x