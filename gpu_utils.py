use_gpu=True
if use_gpu:
    from torch.cuda import FloatTensor, IntTensor
    def to_gpu(x):
        return x.cuda()
else:
    from torch import FloatTensor, IntTensor
    def to_gpu(x):
        return x.cpu()

# the below function is from the Pytorch forums
# https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3
import subprocess
def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map