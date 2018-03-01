import torch
from torch.autograd import Variable
from gpu_utils import get_gpu_memory_map
from visdom_helper.visdom_helper import Dashboard
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
        use_visdom = False,
        dashboard = 'My dashboard',
        ignore_initial=10):
    best_valid_loss = float('inf')
    if use_visdom:
        vis = Dashboard(dashboard)
    plot_counter = 0


    for epoch in range(epochs):
        print('epoch ', epoch)
        scheduler.step()
        for train, data_gen in [True, train_gen], [False, valid_gen]:
            loss_ = 0
            count_ = 0
            if train:
                model.train()
            else:
                model.eval()

            for inputs_, targets_ in data_gen:
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
                if train:
                    line_name = 'training_loss'
                    print('train:',loss_/count_, count_, get_gpu_memory_map())
                else:
                    valid_loss = loss_/count_
                    line_name = 'validation_loss'
                    print('valid:',loss_/count_, count_, get_gpu_memory_map())
                    if count_ > 50:
                        break
                if use_visdom and plot_counter>ignore_initial:
                    try:
                        vis.append(line_name,
                               'line',
                               X=np.array([plot_counter]),
                               Y=np.array([this_loss]))
                    except:
                        print('Please start a visdom server with python -m visdom.server!')
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print("we're improving!", best_valid_loss)
            # spell_out:
            torch.save(model.state_dict(), save_path)
            print("successfully saved model")

        if valid_loss < 1e-10:
            break