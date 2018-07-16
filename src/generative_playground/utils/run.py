import torch
from torch.optim import lr_scheduler
from gpu_utils import to_gpu
from models.simple_models import Net
from torch.utils.data import DataLoader
from data_utils.data_sources import data_gen,DatasetFromModel
from fit import fit

true_w = to_gpu(torch.ones((20, 1)))
random_w = torch.randn(true_w.shape)

true_model = to_gpu(Net(true_w))
model = to_gpu(Net(random_w))

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
criterion = torch.nn.MSELoss()
epochs = 20
save_path = 'test.mdl'

valid_dataset = DatasetFromModel(first_dim=1,batches=256,model=true_model)
train_dataset = DatasetFromModel(first_dim=1,batches=1024,model=true_model)
valid_loader = DataLoader(valid_dataset, batch_size = 256)
train_loader = DataLoader(valid_dataset, batch_size = 64)

fit(train_gen=train_loader,
    valid_gen=valid_loader,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=epochs,
    loss_fn=criterion,
    save_path=save_path)

# now try creating a new model and loading the old weights
model_2 = to_gpu(Net(torch.randn(true_w.shape)))
model_2.load_state_dict(torch.load(save_path))
print(model_2.w)


