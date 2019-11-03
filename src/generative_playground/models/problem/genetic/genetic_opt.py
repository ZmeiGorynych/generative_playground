import random
import datetime
import math
import glob, gzip, pickle
import os
from uuid import uuid4
import numpy as np
from collections import OrderedDict
from generative_playground.models.pg_runner import PolicyGradientRunner

def populate_data_cache(snapshot_dir, reward_cache={}):
    """
    Query the file system for all snapshots so far along with their max rewards
    :param snapshot_dir: data location
    :return: dict {root_name: max_reward}
    """
    files = glob.glob(os.path.realpath(snapshot_dir) + '/*_metrics.zip')
    for file in files:
        file_root = os.path.split(file)[-1].replace('_metrics.zip', '')
        runner_file = os.path.realpath(snapshot_dir) + '/' + file_root +'_runner.zip'
        if os.path.isfile(runner_file):
            if file_root not in reward_cache:
                with gzip.open(file, 'rb') as f:
                    data = pickle.load(f)
                reward_cache[file_root] = {'best_rewards': data['best_reward'].values}
    return reward_cache

def load_coeff_vector_cache(snapshot_dir, coeff_vector_cache):
    """
        Query the file system for all snapshots so far along with their max rewards
        :param snapshot_dir: data location
        :return: dict {root_name: max_reward}
        """
    files = glob.glob(os.path.realpath(snapshot_dir) + '/*_runner.zip')
    for file in files:
        file_root = os.path.split(file)[-1].replace('_runner.zip', '')
        if file_root not in coeff_vector_cache:
            model = PolicyGradientRunner.load(file)
            coeff_vector_cache[file_root] = {'params': model.params}
    return coeff_vector_cache

def get_mean_top_prctile(x, fraction=0.2):
    sorted_x = np.array(sorted(x, reverse=True))
    max_ind = math.ceil(len(x)*fraction)
    out = sorted_x[:max_ind].mean()
    return out

def extract_best(data_cache, num_best, key_fun=get_mean_top_prctile):
    sorted_items = sorted(list(data_cache.items()), reverse=True, key=lambda x: key_fun(x[1]['best_rewards']))
    data_cache_best = OrderedDict(sorted_items[:num_best])
    return data_cache_best


def pick_model_to_run(data_cache, model_class, save_location, num_best=10):
    """
    Pick the next model to simulate
    :param data_cache: dict {root_name: max_reward}
    :return: a ready-to-run model simulator
    """
    data_cache_best = extract_best(data_cache, num_best)
    done = False
    while not done:  # a loop to catch file contention between workers
        try:
            chosen_root = random.choice(list(data_cache_best.keys()))
            model = model_class.load_from_root_name(save_location, chosen_root)
            done = True
        except Exception as e:
            print(e)

    return model


def pick_model_for_crossover(data_cache, model, model_class, save_location, num_best=10):
    """
        Pick the model to cross with the current one
        :param data_cache: dict {root_name: max_reward}
        :param model: the base 'model'
        :return: a ready-to-run model simulator
        """
    filtered_cache = {key: value for key, value in data_cache.items() if key != model.root_name}
    if not filtered_cache:
        raise ValueError("Not enough items in data cache")
    out = pick_model_to_run(filtered_cache, model_class, save_location, num_best)
    return out


def generate_root_name(old_name, data_cache):
    """
    Return a new name
    :param old_name: a string, either arbitrary or 'root_name#timestamp#lineage#uuid'
    :param data_cache:
    :return:
    """
    new_uuid = str(uuid4())
    if '#' in old_name:
        old_root, old_timestamp, old_lineage, old_uuid = old_name.split('#')
    else: #first call
        old_root, old_timestamp, old_lineage, old_uuid = old_name, '', '', ''
    cache_lineages = [x.split('#')[2] for x in data_cache.keys()]
    tmp = old_lineage
    done = False
    letters = 'abcdefghijklmnopqrstuvwxyz'
    while not done:
        for letter in letters:
            lineage = tmp + letter
            if lineage not in cache_lineages:
                done = True
                break
        if not done:
            tmp += 'Z'
    timestamp_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_name = '#'.join([old_root, timestamp_string, lineage, new_uuid])
    return final_name










