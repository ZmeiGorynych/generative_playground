from .policy import PolicyFromTarget

class RePlayer: # to be an environment interface
    def __init__(self):
        self._policy = None

    def update(self, runs):
        self.t = 0
        self._policy = PolicyFromTarget(runs['actions'])
        self.env_out = runs['env_outputs']

    @property
    def policy(self):
        return lambda x: self._policy(x)

    def step(self):
        self.t += 1
        if self.t >= len(self.env_out):
            raise StopIteration
        else:
            return self.env_out[self.t]

def weighted_log_loss(logp, weights):
    pass

def clipped_log_loss(logp, logp_old, eps, weights):
    # weights are the advantage calculation
    pass