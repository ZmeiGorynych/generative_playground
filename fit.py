import torch
from torch.autograd import Variable
from generative_playground.gpu_utils import get_gpu_memory_map

try:
    from generative_playground.visdom_helper.visdom_helper import Dashboard
    have_visdom = True
except:
    have_visdom = False
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
        dashboard = None,
        plot_ignore_initial=0,
        save_always=False,
        exp_smooth = 0.9,
        batches_to_valid=9,
        valid_batches_to_checkpoint = 10,
        grad_clip = None,
        plot_prefix = '',
        loss_display_cap = 4):

    best_valid_loss = float('inf')
    cum_val_loss = 0
    val_count = 0

    if dashboard is not None and have_visdom:
        vis = Dashboard(dashboard)

    plot_counter = 0
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
                loss_name = plot_prefix + ' train_loss'
            else:
                with torch.no_grad():
                    train = False
                    data_iter = valid_iter
                    model.eval()
                    loss_fn.eval()
                    loss_name = plot_prefix + ' val_loss'

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

            try:
                model.reset_hidden()
            except:
                pass

            # show intermediate results
            gpu_usage = get_gpu_memory_map()
            print(loss_name, this_loss, n, gpu_usage )
            plot_counter += 1
            if dashboard is not None and plot_counter > plot_ignore_initial and have_visdom:
                try:
                    vis.append('gpu_usage',
                               'line',
                               X=np.array([plot_counter]),
                               Y=np.array([gpu_usage[0]]))
                    vis.append(loss_name,
                           'line',
                           X=np.array([plot_counter]),
                           Y=np.array([min(loss_display_cap, this_loss)]))
                    if hasattr(loss_fn,'metrics'):
                        vis.append(loss_name + ' metrics',
                                   'line',
                                   X=np.array([plot_counter]),
                                   Y=np.array([[val for key, val in loss_fn.metrics.items()]]),
                                   opts={'legend': [key for key, val in loss_fn.metrics.items()]})
                except Exception as e:
                    print(e)
                    visdom_exists = False
            if train:
                yield this_loss

def save_model(model, save_path = 'insurance.mdl'):
    torch.save(model.state_dict(), save_path)
    print("successfully saved model")