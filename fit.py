import torch
from torch.autograd import Variable
from utils import to_gpu


# true_w = torch.ones((20, 1))
#
# model = to_gpu(Net(true_w.shape))
# optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# criterion = torch.nn.MSELoss()
# epochs = 10
#
# # for now generate the whole datasets and keep them in memory
# valid_gen = data_gen(batch_size=256, batches=1, w=true_w)
# valid_data = next(valid_gen)
#
# train_gen = data_gen(batch_size=64, batches=10, w=true_w)
# train_data = [x for x in train_gen]


def fit(train_gen = None,
        valid_gen = None,
        model = None,
        optimizer = None,
        scheduler = None,
        epochs = None,
        criterion = None,
        save_path = None):
    best_valid_loss = float('inf')
    train_data = train_gen
    valid_data = valid_gen

    for epoch in range(epochs):
        print('epoch ', epoch)
        scheduler.step()
        for train, data in [True, train_data], [False, valid_data]:
            loss_ = 0
            count_ = 0
            for inputs, labels in data:
                try: #if inputs are Tensors, cast them to Variables
                    inputs = Variable(inputs)
                    labels = Variable(labels)
                except:
                    pass
                #print(inputs.shape)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_+=loss.data[0]
                count_+=1
            if train:
                print(loss_/count_, count_)
            else: 
                valid_loss = loss_/count_
                
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print("we're improving!", best_valid_loss)
            # spell_out:
            torch.save(model.state_dict(), save_path)
            print("successfully saved model")

        if valid_loss < 1e-10:
            break