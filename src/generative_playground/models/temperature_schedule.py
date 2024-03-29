import math

def pulsing_schedule(x, period=300, max=10):
    x = (x % period) - period
    x = (x*x) / (period*period) + 1
    return max*x


def toothy_exp_schedule(x, scale=150, max_value=10):
    modifier = 1 - ((x%scale)/scale) # see-saws from 1 to almost 0 then jumps back # used to go from 1 to -1
    temp = max_value**modifier
    return temp


def seesaw_exp_schedule(x, scale=150, max_value=10):
    scale = 2*scale
    modifier = abs(2*(x%scale)/scale - 1)  # see-saws between 0 and 1
    temp = max_value**modifier
    return temp

def reverse_toothy_exp_schedule(x, scale=100, max_value=10):
    modifier = (x%scale)/scale # see-saws from 0 to almost 1 then jumps back # used to go from 1 to -1
    temp = max_value**modifier
    return temp

def shifted_cosine_schedule(x, period=200):
    x1 = (x % period)/period # goes from 0 to almost 1
    x2 = 0.5*(math.cos(math.pi*x1) + 1)
    return x2


class TemperatureCallback:
    def __init__(self, policy, temperature_function):
        self.policy = policy
        self.counter = 0
        self.temp_fun = temperature_function

    def __call__(self, inputs, model, outputs, loss_fn, loss):
        self.counter += 1
        target_temp = self.temp_fun(self.counter)
        self.policy.set_temperature(target_temp)