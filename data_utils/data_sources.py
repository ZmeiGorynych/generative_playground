import torch
from torch.autograd import Variable
from gpu_utils import to_gpu
from torch.utils.data.dataset import Dataset

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
        
class DatasetFromModel(Dataset):
    """Dataset creating data on the fly"""

    def __init__(self, first_dim, batches, model):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.batches = batches
        self.model = model
        self.x_shape = first_dim, model.input_shape[1]
        

    def __len__(self):
        return self.batches

    def __getitem__(self, idx):
        x = to_gpu(torch.randn(self.x_shape))
        y = to_gpu(self.model(Variable(x)).data)
        return (x, y)
        