import torch
from torch.autograd import Variable
from basic_pytorch.gpu_utils import get_gpu_memory_map
from basic_pytorch.visdom_helper.visdom_helper import Dashboard
import numpy as np

def to_variable(x):
    if type(x)==tuple:
        return tuple([to_variable(xi) for xi in x])
    elif 'Variable' not in str(type(x)):
        return Variable(x)
    else:
        return x


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
        exp_smooth = 0.9,
        batches_to_valid=10,
        grad_clip = None):

    best_valid_loss = float('inf')

    if dashboard is not None:
        vis = Dashboard(dashboard)
    plot_counter = 0

    valid_batches = max(1,int(batches_to_valid*len(valid_gen)/len(train_gen)))

    for epoch in range(epochs):
        print('epoch ', epoch)
        scheduler.step()
        train_iter = train_gen.__iter__()
        valid_iter = valid_gen.__iter__()
        done={True:False,False:False}
        val_loss = 0
        val_count = 0
        for n in range(len(train_gen) + len(valid_gen)):
            if n%(batches_to_valid + valid_batches) <batches_to_valid:
                train = True
                data_iter = train_iter
                model.train()
                loss_fn.train()
                loss_name = 'training_loss'
            else:
                train = False
                data_iter = valid_iter
                model.eval()
                loss_fn.eval()
                loss_name = 'validation_loss'

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
            this_loss = loss.data[0]
            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm(model.parameters(), grad_clip)
                optimizer.step()
            else:
                val_loss += this_loss
                val_count += 1
                # # TODO: remove this!
                # save_model(model, save_path)
            try:
                model.reset_hidden()
            except:
                pass

            # show intermediate results
            print(loss_name, this_loss, n, get_gpu_memory_map())
            plot_counter += 1
            if dashboard is not None and plot_counter>plot_ignore_initial:
                try:
                    vis.append(loss_name,
                           'line',
                           X=np.array([plot_counter]),
                           Y=np.array([min(1.0, this_loss)]))
                    if hasattr(loss_fn,'metrics'):
                        vis.append(loss_name + ' metrics',
                                   'line',
                                   X=np.array([plot_counter]),
                                   Y=np.array([[min(val,1.0) for key, val in loss_fn.metrics.items()]]),
                                   opts={'legend': [key for key, val in loss_fn.metrics.items()]})
                except:
                    print('Please start a visdom server with python -m visdom.server!')
        valid_loss = val_loss / val_count
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print("we're improving!", best_valid_loss)
            # spell_out:
            save_model(model,save_path)

        if valid_loss < 1e-10:
            break

def save_model(model, save_path = 'insurance.mdl'):
    torch.save(model.state_dict(), save_path)
    print("successfully saved model")