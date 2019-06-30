import matplotlib.pyplot as plt

from gridworld.agent import Agent
from gridworld.environment import Gridworld


if __name__ == '__main__':
    agent = Agent(
        Gridworld.deterministic_easy(),
        epsilon=0.1,
        gamma=0.9,
    )
    rewards = agent.train(100)
    rewards.expanding().mean().plot()
    plt.show()
