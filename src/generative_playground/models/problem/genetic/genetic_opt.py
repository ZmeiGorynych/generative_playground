import random
import datetime
import glob, gzip, pickle
import os
from uuid import uuid4




def populate_data_cache(snapshot_dir, reward_cache={}):
    """
    Query the file system for all snapshots so far along with their max rewards
    :param snapshot_dir: data location
    :return: dict {root_name: max_reward}
    """
    #TODO: check that the runner save file exists too!
    files = glob.glob(os.path.realpath(snapshot_dir) + '/*_metrics.zip')
    for file in files:
        file_root = os.path.split(file)[-1].replace('_metrics.zip', '')
        if file_root not in reward_cache:
            with gzip.open(file, 'rb') as f:
                data = pickle.load(f)
            best_reward = data['best_reward'].max()
            reward_cache[file_root] = best_reward
    return reward_cache


def pick_model_to_run(data_cache, model_factory, model_class, save_location, num_best=10):
    """
    Pick the next model to simulate
    :param data_cache: dict {root_name: max_reward}
    :return: a ready-to-run model simulator
    """
    if not data_cache:
        return model_factory()
    else:
        sorted_items = sorted(list(data_cache.items()), reverse=True, key=lambda x: x[1])
        data_cache_best = dict(sorted_items[:num_best])
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
    out = pick_model_to_run(filtered_cache, None, model_class, save_location, num_best)
    return out


def generate_root_name(old_name, data_cache):
    """
    Return a new name
    :param old_name: a string, either arbitrary or 'root_name#timestamp#lineage#uuid'
    :param data_cache:
    :return:
    """
    new_uuid = str(uuid4())
    if len(old_name) > len(new_uuid):
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










