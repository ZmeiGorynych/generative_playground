import torch

# fit_dataset = ImagesDataset(path_to_fit_images)
# fit_data_loader = torch.utils.data.DataLoader(fitdataset, batch_size=10)
#
# valid_dataset = ImagesDataset(path_to_valid_images)
# valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=10)


# class ImagesDataset(torch.utils.data.Dataset):
#     pass


def data_gen(batch_size, batches, w):
    for _ in range(batches):
        x = torch.randn(batch_size, w.shape[0])
        y = x @ w
        yield x, y