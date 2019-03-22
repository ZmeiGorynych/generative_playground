from generative_playground.models.problem.rl.base_task import BaseTask
from generative_playground.models.problem.rl.environment import SequenceEnvironment

class SequenceGenerationTask(BaseTask):
    def __init__(self, name='seq_gen',
                 molecules=True,
                 grammar= False,
                 reward_fun = None,
                 batch_size = 1,
                 log_dir=None,
                 max_steps=None,
                 save_dataset=None,
                 grammar_cache=None):
        super().__init__()
        self.name = name
        self.env = SequenceEnvironment(molecules,
                                       grammar,
                                       reward_fun=reward_fun,
                                       batch_size=batch_size,
                                       max_steps=max_steps,
                                       save_dataset=save_dataset,
                                       grammar_cache=grammar_cache)
        self.action_dim = self.env.action_dim
        self.state_dim = self.env.state_dim
        #self.env = self.set_monitor(self.env, log_dir)