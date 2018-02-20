import torch
from torch.autograd import Variable

from gpu_utils import to_gpu

def inputs2pytorch(inputs_):
    # reshape the data to our needs
    inputs = torch.FloatTensor(inputs_['the_input']).permute(1, 0, 2)
    inputs = Variable(to_gpu(inputs), requires_grad=True)

    probs_sizes = inputs_['input_length'].T[0]
    label_sizes = inputs_['label_length'].T[0]
    labels = []
    for i, row in enumerate(inputs_['the_labels']):
        labels += list(row[:int(label_sizes[i])])

    labels = Variable(torch.IntTensor([int(label) for label in labels]))
    probs_sizes = Variable(torch.IntTensor([int(x) for x in probs_sizes]))
    label_sizes = Variable(torch.IntTensor([int(x) for x in label_sizes]))
    return (inputs,probs_sizes), (labels,label_sizes)