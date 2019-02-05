import torch
import numpy as np
from collections import OrderedDict

def language_metrics_for_monitor(inputs, targets, model_out, train, plot_counter, index_to_lang=None, smooth=0.9):
    out = {}
    if not train:
        pre_metrics = get_language_metrics(model_out, targets, index_to_lang)
        for label, target in targets.items():
            out[label + " validation breakdown"] = {'type': 'line',
                          'X': np.array([plot_counter]),
                          'Y': np.array([[float('nan') if lang not in pre_metrics[label] else pre_metrics[label][lang] for lang in index_to_lang.values()]]),
                          'opts': {'legend': [lang for lang in index_to_lang.values()]},
                                                    'smooth': smooth}

    return out

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