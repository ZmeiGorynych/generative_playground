from generative_playground.models.problem.rl.environment import SequenceEnvironment

class BaseTask:
    def __init__(self):
        pass

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        # TODO: really the environment should be in control, say max_seq_len, etc?
        # if done:
        #     next_state = self.env.reset()
        return next_state, reward, done, info

    def seed(self, random_seed):
        return self.env.seed(random_seed)

class SequenceGenerationTask(BaseTask):
    def __init__(self, name='seq_gen',
                 molecules=True,
                 grammar= False,
                 reward_fun = None,
                 batch_size = 1,
                 log_dir=None,
                 max_steps=None,
                 save_dataset=None):
        super().__init__()
        self.name = name
        self.env = SequenceEnvironment(molecules,
                                       grammar,
                                       reward_fun=reward_fun,
                                       batch_size=batch_size,
                                       max_steps=max_steps,
                                       save_dataset=save_dataset)
        self.action_dim = self.env.action_dim
        self.state_dim = self.env.state_dim

class SequenceGenerationTask(BaseTask):
    def __init__(self, name='seq_gen',
                 molecules=True,
                 grammar=False,
                 reward_fun=None,
                 batch_size=1,
                 log_dir=None,
                 max_steps=None,
                 save_dataset=None):
        super().__init__()
        self.name = name
        self.env = SequenceEnvironment(molecules,
                                       grammar,
                                       reward_fun=reward_fun,
                                       batch_size=batch_size,
                                       max_steps=max_steps,
                                       save_dataset=save_dataset)
        self.action_dim = self.env.action_dim
        self.state_dim = self.env.state_dim
        #self.env = self.set_monitor(self.env, log_dir)