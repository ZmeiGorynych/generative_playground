import torch
from torch.autograd import Variable
from generative_playground.gpu_utils import get_gpu_memory_map


import numpy as np

def to_variable(x):
    if type(x)==tuple:
        return tuple([to_variable(xi) for xi in x])
    elif 'Variable' not in str(type(x)):
        return Variable(x)
    else:
        return x

# The fit function is a generator, so one can call several of these in
# the sequence one desires
def fit(train_gen = None,
        valid_gen = None,
        model = None,
        optimizer = None,
        scheduler = None,
        epochs = None,
        loss_fn = None,
        save_path = None,
        save_always=False,
        batches_to_valid=9,
        valid_batches_to_checkpoint = 10,
        grad_clip = None,
        plot_prefix='',
        loss_display_cap=4,
        dashboard_name=None,
        plot_ignore_initial=0
        ):

    metric_plotter = MetricPlotter(plot_prefix=plot_prefix,
                                   loss_display_cap=loss_display_cap,
                                   dashboard_name=dashboard_name,
                                   plot_ignore_initial=plot_ignore_initial)

    best_valid_loss = float('inf')
    cum_val_loss = 0
    val_count = 0

    valid_batches = max(1,int(batches_to_valid*len(valid_gen)/len(train_gen)))
    if 'ReduceLROnPlateau' in str(type(scheduler)):
        step_scheduler_after_val = True

    for epoch in range(epochs):
        print('epoch ', epoch)
        if not step_scheduler_after_val:
            scheduler.step()
        train_iter = train_gen.__iter__()
        valid_iter = valid_gen.__iter__()
        done={True:False,False:False}

        for n in range(len(train_gen) + len(valid_gen)):
            if n%(batches_to_valid + valid_batches) <batches_to_valid:
                train = True
                data_iter = train_iter
                model.train()
                loss_fn.train()
            else:
                with torch.no_grad():
                    train = False
                    data_iter = valid_iter
                    model.eval()
                    loss_fn.eval()

            # get the next pair (inputs, targets)
            try:
                inputs_, targets_ = next(data_iter)
            except StopIteration:
                # make sure we get all data from both iterators
                done[train] = True
                if done[True] and done[False]:
                    break
                else:
                    continue

            inputs = to_variable(inputs_)
            targets = to_variable(targets_)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            this_loss = loss.data.item()
            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    nice_params = filter(lambda p: p.requires_grad, model.parameters())
                    torch.nn.utils.clip_grad_norm(nice_params, grad_clip)
                optimizer.step()
            else:
                cum_val_loss += this_loss
                val_count += 1
                # after enough validation batches, see if we want to save the weights
                if val_count >= valid_batches_to_checkpoint:
                    valid_loss = cum_val_loss / val_count
                    if step_scheduler_after_val:
                        scheduler.step(valid_loss)
                    val_count = 0
                    cum_val_loss = 0
                    if valid_loss < best_valid_loss or save_always:
                        if valid_loss < best_valid_loss:
                            best_valid_loss = valid_loss
                            print("we're improving!", best_valid_loss)
                        # spell_out:
                        if save_path is not None:
                            save_model(model, save_path)

            # try:
            #     model.reset_hidden()
            # except:
            #     pass
            # plot the intermediate metrics
            metric_plotter(train, this_loss, loss_fn.metrics if hasattr(loss_fn, 'metrics') else None)

            if train:
                yield this_loss

class MetricPlotter:
    def __init__(self,
                 plot_prefix = '',
                 loss_display_cap = 4,
                 dashboard_name=None,
                 plot_ignore_initial=0):
        self.plot_prefix = plot_prefix
        self.plot_ignore_initial = plot_ignore_initial
        self.loss_display_cap = loss_display_cap
        self.plot_counter = 0
        try:
            from generative_playground.visdom_helper.visdom_helper import Dashboard
            if dashboard_name is not None:
                self.vis = Dashboard(dashboard_name)
                self.have_visdom = True
        except:
            self.have_visdom = False
            self.vis = None

    def __call__(self, train, this_loss, metrics):
        '''
        Plot the results of the latest batch
        :param train: bool: was this a traning batch?
        :param this_loss: float: latest loss
        :param metrics: dict {str:float} with any additional metrics
        :return: None
        '''
        if not self.have_visdom:
            return

        if train:
            loss_name = self.plot_prefix + ' train_loss'
        else:
            loss_name = self.plot_prefix + ' val_loss'

        # show intermediate results
        gpu_usage = get_gpu_memory_map()
        print(loss_name, this_loss, self.plot_counter, gpu_usage)
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
                           Y=np.array([min(self.loss_display_cap, this_loss)]))
                if metrics is not None:
                    self.vis.append(loss_name + ' metrics',
                               'line',
                               X=np.array([self.plot_counter]),
                               Y=np.array([[val for key, val in metrics.items()]]),
                               opts={'legend': [key for key, val in metrics.items()]})
            # except Exception as e:
            #     print(e)
            #     visdom_exists = False

def save_model(model, save_path = 'insurance.mdl'):
    torch.save(model.state_dict(), save_path)
    print("successfully saved model")