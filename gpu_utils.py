use_gpu=True
if use_gpu:
    from torch.cuda import FloatTensor, IntTensor
    def to_gpu(x):
        return x.cuda()
else:
    from torch import FloatTensor, IntTensor
    def to_gpu(x):
        return x.cpu()
    