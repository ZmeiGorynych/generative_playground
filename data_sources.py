import torch
from torch.autograd import Variable
from utils import to_gpu

# fit_dataset = ImagesDataset(path_to_fit_images)
# fit_data_loader = torch.utils.data.DataLoader(fitdataset, batch_size=10)
#
# valid_dataset = ImagesDataset(path_to_valid_images)
# valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=10)


# class ImagesDataset(torch.utils.data.Dataset):
#     pass


def data_gen(batch_size, batches, model):
    for _ in range(batches):
        x = to_gpu(Variable(torch.randn(batch_size, model.input_shape[1])))
        y = to_gpu(model.forward(x))
        y = y.detach()
        yield x, y
        
