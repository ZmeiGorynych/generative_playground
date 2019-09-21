import random
import networkx as nx
import glob, gzip, pickle
import os
# only runs one process at a time, for simplicity
from generative_playground.models.problem.genetic.crossover import mutate, crossover

snapshot_dir = '.'
top_N = 10
p_mutate = 0.2
p_crossover = 0.2
relationships = nx.DiGraph()

def populate_data_cache(snapshot_dir, reward_cache={}):
    """
    Query the file system for all snapshots so far along with their max rewards
    :param snapshot_dir: data location
    :return: dict {root_name: max_reward}
    """
    files = glob.glob(os.path.realpath(snapshot_dir) + '/*_metrics.zip')
    for file in files:
        file_root = os.path.split(file)[-1].replace('_metrics.zip', '')
        if file_root not in reward_cache:
            with gzip.open(file, 'rb') as f:
                data = pickle.load(f)
            best_reward = data['best_reward'].max()
            reward_cache[file_root] = best_reward
    return reward_cache


def pick_model_to_run(data_cache, model_factory, model_class, num_best=10):
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
        chosen_root = random.choice(list(data_cache_best.keys()))
        model = model_class.load_from_root_name(chosen_root)
        return model

def pick_model_for_crossover(data_cache, model, model_class, num_best=10):
    """
        Pick the model to cross with the current one
        :param data_cache: dict {root_name: max_reward}
        :param model: the base 'model'
        :return: a ready-to-run model simulator
        """
    filtered_cache = {key: value for key, value in data_cache.items() if key != model.root_name}
    if not filtered_cache:
        raise ValueError("Not enough items in data cache")
    out = pick_model_to_run(filtered_cache, None, model_class, num_best)
    return out


def generate_root_name(old_name, data_cache):
    """
    Return a new name
    :param old_name:
    :param data_cache:
    :return:
    """
    tmp = old_name
    done = False
    letters = ['abcdefghijklmnopqrstuvwxyz']
    while not done:
        for letter in letters:
            name = tmp + letter
            if name not in data_cache:
                done = True
                break
        if not done:
            tmp += 'Z'
    return name #TODO: is this the correct one?

if __name__ == '__main__':
    data_cache = {}
    while True:
        data_cache = populate_data_cache(snapshot_dir, data_cache)
        # sort data cache by reward, only keep top_N values
        model = pick_model_to_run(data_cache, model_factory, PolicyGradientRunner)
        orig_name = model.root_name
        model.set_root_name(generate_root_name(orig_name, data_cache))
        relationships.add_edge(orig_name, model.root_name)

        test = random.random()
        if test < p_mutate:
            model = mutate(model)
        elif test < p_mutate + p_crossover and len(data_cache) > 1:
            second_model = pick_model_for_crossover(data_cache, model)
            model = crossover(model, second_model)
            relationships.add_edge(second_model.root_name, model.root_name)

        # model.rebase() # make the conditional probs part of the new priors?
        model.run()
        # TODO: save relationships graph










