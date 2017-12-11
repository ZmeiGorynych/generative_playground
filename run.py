import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from utils import to_gpu
from models import Net
from data import data_gen
from fit import fit

true_w = torch.ones((20, 1))

model = to_gpu(Net(true_w.shape))
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
criterion = torch.nn.MSELoss()
epochs = 10
save_path = 'test.mdl'

# for now generate the whole datasets and keep them in memory
valid_gen = data_gen(batch_size=256, batches=1, w=true_w)
valid_data = next(valid_gen)

train_gen = data_gen(batch_size=64, batches=10, w=true_w)
train_data = [x for x in train_gen]


fit(train_data = train_data,
        valid_data = valid_data,
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        epochs = epochs,
        criterion = criterion,
        save_path=save_path)

# now try creating a new model and loading the old weights
model_2 = to_gpu(Net(true_w.shape))
model_2.load_state_dict(torch.load(save_path))
print(model_2.w)


