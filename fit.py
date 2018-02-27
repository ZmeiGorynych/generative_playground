import torch
from torch.autograd import Variable
from gpu_utils import get_gpu_memory_map

def fit(train_gen = None,
        valid_gen = None,
        model = None,
        optimizer = None,
        scheduler = None,
        epochs = None,
        loss_fn = None,
        save_path = None):
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        print('epoch ', epoch)
        scheduler.step()
        for train, data_gen in [True, train_gen], [False, valid_gen]:
            loss_ = 0
            count_ = 0
            # data_gen is a FUNCTION that returns a new generator of batches for each epoch
            print('starting next epoch...')
            # try:
            #     del outputs, loss, labels, inputs, my_data
            # except:
            #     pass
            my_data = data_gen()
            for inputs_, targets_ in data_gen():
                inputs = inputs_#Variable(inputs_)
                targets = targets_#Variable(targets_)
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

                loss_+= loss.data[0]
                count_+= 1
                if train:
                    print('train:',loss_/count_, count_, get_gpu_memory_map())
                else:
                    valid_loss = loss_/count_
                    print('valid:',loss_/count_, count_, get_gpu_memory_map())
                    if count_ > 10:
                        break

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print("we're improving!", best_valid_loss)
            # spell_out:
            torch.save(model.state_dict(), save_path)
            print("successfully saved model")

        if valid_loss < 1e-10:
            break