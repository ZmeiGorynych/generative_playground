from deep_rl.utils.misc import random_seed
import numpy as np
import pickle
from generative_playground.utils.gpu_utils import get_gpu_memory_map

def run_iterations(agent, visdom = None, invalid_value = None):
    if visdom is not None:
        have_visdom = True
    random_seed()
    config = agent.config
    agent_name = agent.__class__.__name__
    iteration = 0
    steps = []
    rewards = []
    sm_metrics = np.array([[invalid_value,invalid_value,invalid_value]])
    while True:
        agent.iteration()
        steps.append(agent.total_steps)
        rewards.append(np.mean(agent.last_episode_rewards))
        if iteration % config.iteration_log_interval == 0:
            config.logger.info('total steps %d, mean/max/min reward %f/%f/%f' % (
                agent.total_steps, np.mean(agent.last_episode_rewards),
                np.max(agent.last_episode_rewards),
                np.min(agent.last_episode_rewards)
            ))
            print('total steps %d, mean/max/min reward %f/%f/%f' % (
                agent.total_steps, np.mean(agent.last_episode_rewards),
                np.max(agent.last_episode_rewards),
                np.min(agent.last_episode_rewards)
            ))

        if have_visdom:
            try:
                reward = agent.last_episode_rewards
                #reward = reward[reward>0]
                if len(reward)>0:
                    metrics = np.array([[len(reward[reward!=invalid_value])/100, np.mean(reward), np.max(reward)]])
                else:
                    metrics = np.array([[invalid_value,invalid_value,invalid_value]])
                sm_metrics = 0.95*sm_metrics + 0.05*metrics
                visdom.append('molecule validity',
                              'line',
                              X=np.array([iteration]),
                              Y=sm_metrics,#np.array([sm_metrics]),
                              opts={'legend': ['num_valid', 'avg_len', 'max_len']})
                visdom.append('gpu usage',
                              'line',
                              X=np.array([iteration]),
                              Y=np.array([get_gpu_memory_map()[0]]),
                              opts={'legend': ['gpu']})
            except Exception as e:
                print(e)

                #have_visdom = False

        if False and iteration % (config.iteration_log_interval * 100) == 0:
            with open('data/%s-%s-online-stats-%s.bin' % (agent_name, config.tag, agent.task.name), 'wb') as f:
                pickle.dump({'rewards': rewards,
                             'steps': steps}, f)
                print({'rewards': rewards, 'steps': steps})
            agent.save('data/%s-%s-model-%s.bin' % (agent_name, config.tag, agent.task.name))
        iteration += 1
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break

    return steps, rewards
