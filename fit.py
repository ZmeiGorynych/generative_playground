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
        plot_ignore_initial=10,
        exp_smooth = 0.9,
        batches_to_valid=10):

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
        for n in range(len(train_gen) + len(valid_gen)):
            if n%(batches_to_valid + valid_batches) <batches_to_valid:
                train = True
                data_iter = train_iter
                model.train()
                loss_name = 'training_loss'
            else:
                train = False
                data_iter = valid_iter
                model.eval()
                loss_name = 'validation_loss'

            loss_ = 0
            count_ = 0

            # get the next pair (inputs, targets)
            try:
                inputs_, targets_ = next(data_iter)
            except StopIteration:
                done[train] = True
                if done[True] and done[False]:
                    break
                else:
                    continue

            inputs = to_variable(inputs_)
            targets = to_variable(targets_)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            try:
                model.reset_hidden()
            except:
                pass
            this_loss = loss.data[0]
            loss_+= this_loss
            count_+= 1
            plot_counter += 1
            if not train:
                valid_loss = loss_/count_
                if count_ > 50:
                    break
            # elif save_every and count_>0 and count_%save_every==0:
            #     save_model(model)
            # show intermediate results
            print(loss_name, loss_ / count_, n, get_gpu_memory_map())
            if dashboard is not None and plot_counter>plot_ignore_initial:
                try:
                    vis.append(loss_name,
                           'line',
                           X=np.array([plot_counter]),
                           Y=np.array([this_loss]))
                except:
                    print('Please start a visdom server with python -m visdom.server!')
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