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


def fit(train_data = None,
        valid_data = None,
        model = None,
        optimizer = None,
        scheduler = None,
        epochs = None,
        criterion = None,
        save_path = None):
    best_valid_loss = torch.cuda.FloatTensor([float('inf')])
    for epoch in range(epochs):
        print('epoch ', epoch)
        # training
        scheduler.step()
        for inputs, labels in train_data:
            inputs = Variable(to_gpu(inputs))
            labels = Variable(to_gpu(labels))

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation:
        valid_inputs, valid_labels = valid_data
        inputs = Variable(to_gpu(valid_inputs))
        labels = Variable(to_gpu(valid_labels))

        outputs = model(inputs)
        loss = criterion(outputs, labels).data
        if loss[0] < best_valid_loss[0]:
            best_valid_loss = loss
            print("we're improving!", best_valid_loss[0])
            # spell_out:
            torch.save(model.state_dict(), save_path)
            print("successfully saved model")

    print(model.w.data)