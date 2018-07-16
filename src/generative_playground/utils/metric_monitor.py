import numpy as np
import copy
import pandas as pd
import pickle, gzip
import datetime
from generative_playground.utils.gpu_utils import get_gpu_memory_map
import os, inspect

class MetricPlotter:
    def __init__(self,
                 plot_prefix = '',
                 save_file = None,
                 loss_display_cap = 4,
                 dashboard_name=None,
                 plot_ignore_initial=0):
        self.plot_prefix = plot_prefix
        if save_file is not None:
            self.save_file = save_file
        else:
            self.save_file = plot_prefix.replace(' ','_') + '.pkz'

        my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        root_location = my_location + '/../'
        self.save_file = root_location + 'train/vae/data/' + self.save_file
        self.plot_ignore_initial = plot_ignore_initial
        self.loss_display_cap = loss_display_cap
        self.plot_counter = 0
        self.stats = pd.DataFrame(columns=['batch', 'timestamp', 'gpu_usage', 'train', 'loss'])
        try:
            from generative_playground.visdom_helper.visdom_helper import Dashboard
            if dashboard_name is not None:
                self.vis = Dashboard(dashboard_name)
                self.have_visdom = True
        except:
            self.have_visdom = False
            self.vis = None

    def __call__(self, train, loss, metrics):
        '''
        Plot the results of the latest batch
        :param train: bool: was this a traning batch?
        :param loss: float: latest loss
        :param metrics: dict {str:float} with any additional metrics
        :return: None
        '''

        if train:
            loss_name = self.plot_prefix + ' train_loss'
        else:
            loss_name = self.plot_prefix + ' val_loss'

        # show intermediate results
        gpu_usage = get_gpu_memory_map()
        print(loss_name, loss, self.plot_counter, gpu_usage)
        self.plot_counter += 1
        if self.vis is not None and self.plot_counter > self.plot_ignore_initial and self.have_visdom:
            #try:
                self.vis.append('gpu_usage',
                           'line',
                           X=np.array([self.plot_counter]),
                           Y=np.array([gpu_usage[0]]))
                self.vis.append(loss_name,
                           'line',
                                X=np.array([self.plot_counter]),
                                Y=np.array([min(self.loss_display_cap, loss)]))
                if metrics is not None:
                    self.vis.append(loss_name + ' metrics',
                               'line',
                               X=np.array([self.plot_counter]),
                               Y=np.array([[val for key, val in metrics.items()]]),
                               opts={'legend': [key for key, val in metrics.items()]})

        metrics =  {} if metrics is None else copy.copy(metrics)
        metrics['train'] = train
        metrics['gpu_usage'] = gpu_usage[0]
        metrics['loss'] = loss
        metrics['batch'] = self.plot_counter
        metrics['timestamp'] = datetime.datetime.now()

        self.stats = self.stats.append(metrics, ignore_index=True)

        if not train: # only save to disk during valdation calls for speedup
            with gzip.open(self.save_file,'wb') as f:
                pickle.dump(self.stats, f)