import torch

def get_language_metrics(model_out, target, index_to_lang):
    acc = {}
    for label, tgt in target.items():
        this_accuracy = accuracy_by_group(model_out[label][:, 1:, :], tgt[:, 1:], tgt[:, 0])
        acc[label] = {index_to_lang[key]: value for key, value in this_accuracy.items()}
    return acc

    # print(acc)
def accuracy_by_group(x, tgt, group, ignore_idx=0):
    out = {}
    for ind in set(group):
        this_x = x[group == ind,:]
        this_tgt = tgt[group == ind, :]
        accurate = (torch.argmax(this_x, dim=2) == this_tgt).to(dtype=torch.float32)
        out[ind.item()] = nonpad_mean(accurate, this_tgt, ignore_idx)
    return out

def nonpad_mean(x, tgt, pad_idx = 0):
    flat_x = x.view(-1)
    flat_tgt = tgt.view(-1)
    nice_x = flat_x[flat_tgt != pad_idx]
    mean = torch.mean(nice_x)
    return mean.item()