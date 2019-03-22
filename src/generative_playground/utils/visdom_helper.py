# original code taken from https://github.com/episodeyang/visdom_helper,
# edited by Egor Kraev egor.kraev@gmail.com to support svg output and a multi-plot input format and batch visdom calls


from visdom import Visdom
import numpy as np
import copy
from frozendict import frozendict

class Dashboard(Visdom):
    def __init__(self,
                 name,
                 server='http://52.213.134.161',
                 call_every=1):
        super(Dashboard, self).__init__(server=server)
        self.env = name
        self.plots = {}
        self.plot_data = {}
        self.call_every = call_every
        self.metric_cache = []

    def plot(self, name, type, *args, **kwargs):
        if 'opts' not in kwargs:
            kwargs['opts'] = {}
        if 'title' not in kwargs['opts']:
            kwargs['opts']['title'] = name

        if hasattr(self, type):
            if name in self.plots:
                getattr(self, type)(win=self.plots[name], *args, **kwargs)
            else:
                id = getattr(self, type)(*args, **kwargs)
                self.plots[name] = id
        else:
            raise AttributeError('plot type: {} does not exist. Please'
                                 'refer to visdom documentation.'.format(type))

    def append(self, name, type, *args, **kwargs):
        if name in self.plots and type != 'svg':
            self.plot(name, type, *args, update='append', **kwargs)
        else:
            self.plot(name, type, *args, **kwargs)

    def _plot_metric_dict(self, metric_dict):
        for title, metric in metric_dict.items():
                self.append(title,
                          metric['type'],
                          **{key:value for key, value in metric.items() if key not in ['type','smooth']})

    def plot_metric_dict(self, metric_dict):
        if len(self.metric_cache) < self.call_every:
            self.metric_cache.append(frozendict({key: frozendict(value) for key, value in metric_dict.items()}))
        else:
            metrics = collapse_metrics(self.metric_cache)
            self.metric_cache = []
            self._plot_metric_dict(metrics)

    def remove(self, name):
        del self.plots[name]

    def clear(self):
        self.plots = {}

def collapse_metrics(metric_cache):
    metric_stack = {}
    # sort them by title
    for metrics in metric_cache:
        for title, metric in metrics.items():
            if title not in metric_stack:
                metric_stack[title] =[]
            metric_stack[title].append(metric)
    out_metrics = {}
    for title, metric_list in metric_stack.items():
        out_metrics[title] = dict(metric_list[-1])
        if out_metrics[title]['type'] == 'line': # batch all the point updates
            out_metrics[title]['X'] = np.concatenate([m['X'] for m in metric_list])
            out_metrics[title]['Y'] = np.concatenate([m['Y'] for m in metric_list])

    return out_metrics












if __name__ == '__main__':
    vis = Dashboard('my-dashboard')
    #vis.plot()
    for i in range(3):
        vis.append('training_loss',
               'line',
               X=np.array([i]),
               Y=np.array([i]))