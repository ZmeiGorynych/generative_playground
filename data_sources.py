import torch

# fit_dataset = ImagesDataset(path_to_fit_images)
# fit_data_loader = torch.utils.data.DataLoader(fitdataset, batch_size=10)
#
# valid_dataset = ImagesDataset(path_to_valid_images)
# valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=10)


# class ImagesDataset(torch.utils.data.Dataset):
#     pass


def data_gen(batch_size, batches, model):
    for _ in range(batches):
        x = torch.randn(batch_size, model.input_size())
        y = model.forward(x)
        yield x, y