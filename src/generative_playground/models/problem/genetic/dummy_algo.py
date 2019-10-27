import numpy as np
import random
from collections import defaultdict


class Model:
    def __init__(self, dimension=30, param_gen=lambda x: np.random.uniform(size=x), params=None):
        if params is None:
            self.params = param_gen(dimension)
            self.value = None
        else:
            self.params = params

class RandomEvaluate:
    """
    Evaluates the current model
    :param model: a model
    :return: an object that describes the model fitness, could be a number or a distribution, implementing __ge__
    """
    def __init__(self):
        self.cache = defaultdict(random.random)

    def __call__(self, model):
        return self.cache[tuple(model.params)]

def better(model1, model2, evaluator):
    """

    :param model1: a model
    :param model2: a model
    :return: one of the two models, depending on the evaluation
    """
    for model in (model1, model2):
        if model.value is None:
            model.value = evaluator(model)
    prob_1_better = float(model1.value > model2.value) + \
                    0.5*float(model1.value == model2.value)
    test = random.random()
    if test > prob_1_better:
        return model2
    else:
        return model1

def tournament_selection(population, evaluator, nt=2):
    inds = [range(nt)]
    pre_selection = [random.choice(population) for _ in range(nt)]
    model = pre_selection.pop(-1)
    while len(pre_selection):
        model = better(model, pre_selection.pop(-1), evaluator)
    return model

def prune(model, parameter_capper):
    model.params = parameter_capper(model.params)
    return model

def mutate(model, delta_scale, parameter_capper):
    delta = np.random.normal(scale=delta_scale, size=model.params.shape)
    model.params += delta
    model = prune(model, parameter_capper)
    return model

def crossover(model1, model2, d, parameter_capper=lambda x: x):
    alpha = np.random.uniform(-d, 1+d)
    child_params = model1.params + alpha*(model2.params - model1.params)
    child = Model(params=child_params)
    child.params = parameter_capper(child.params)
    return child



if __name__ == '__main__':
    N = 100
    dimension = 30
    scale = 1
    md = 0.3
    num_evaluations = 50
    num_iterations = 100000
    for normal in (True, False):
        normal = True
        mean_std = np.zeros(num_evaluations)
        for ie in range(num_evaluations):
            if normal:
                param_gen = lambda n: np.random.normal(size=n, scale=2*scale)
                parameter_capper = lambda x: x # for a normal distribution, don't need to worry about range
                scale_mult = 1.0
            else:
                param_gen = lambda n: np.random.uniform(size=n, high=scale)
                parameter_capper = lambda x: np.minimum(np.maximum(0, x), scale)
                scale_mult = np.sqrt(12)
            population = [Model(dimension, param_gen) for _ in range(N)]
            evaluator = RandomEvaluate()
            for p in population:
                p.value = evaluator(p)
            init_std = np.array([p.params for p in population]).std(axis=0).mean()

            for iter in range(num_iterations):
                parent1 = tournament_selection(population, evaluator)
                parent2 = tournament_selection(population, evaluator)
                child = crossover(parent1, parent2, d=0.25, parameter_capper=parameter_capper)
                child = mutate(child, delta_scale=md*scale, parameter_capper=parameter_capper)
                child.value = evaluator(child)
                if child.value > population[-1].value:
                    population.pop(-1)
                    population.append(child)
                    population = sorted(population, reverse=True, key=lambda x: x.value)
            values = np.array([x.value for x in population])
            coeffs = np.array([x.params for x in population])
            mean_std[ie] = coeffs.std(axis=0).mean()*scale_mult/init_std
        print(mean_std.mean(), mean_std.std())

