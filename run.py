import torch
from torch.optim import lr_scheduler
from gpu_utils import to_gpu
from models.simple_models import Net
from data_utils.data_sources import data_gen,DatasetFromModel
from fit import fit

true_w = to_gpu(torch.ones((20, 1)))

model = to_gpu(Net(true_w.shape))

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
criterion = torch.nn.MSELoss()
epochs = 10
save_path = 'test.mdl'

# for now generate the whole datasets and keep them in memory
true_model = to_gpu(Net(true_w.shape))
true_model.w.data = true_w

# TODO: rewrite this using the Dataset interface
valid_gen = lambda: data_gen(batch_size=256, batches=1, model=true_model)
train_gen = lambda: data_gen(batch_size=64, batches=10, model=true_model)

fit(train_gen = train_gen,
    valid_gen = valid_gen,
    model = model,
    optimizer = optimizer,
    scheduler = scheduler,
    epochs = epochs,
    loss_fn= criterion,
    save_path=save_path)

# now try creating a new model and loading the old weights
model_2 = to_gpu(Net(true_w.shape))
model_2.load_state_dict(torch.load(save_path))
print(model_2.w)


