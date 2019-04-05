from collections import deque

from QLearning.Q_learning import QLearningAgent
from utils import create_uniform_grid, run
from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt

from DQLearning.dqn_agent import Agent

from utils import discretize


def preprocess_state(state, state_grid):
    """Map a continuous state to its discretized representation."""
    return discretize(state, state_grid)


def run_dql(env, dql_agent, state_grid, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        env (object) : environment from with the agent will learn
        agent (object): Deep Q learng agent
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size

    for i_episode in range(1, n_episodes + 1):
        action = np.random.randint(action_size)
        env.reset(train_mode=True)[brain_name]
        env_info = env.step(action)[brain_name]
        state = env_info.vector_observations[0]

        score = 0
        for t in range(max_t):
            action = dql_agent.act(state, eps)

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            dql_agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(dql_agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break

    return scores


if __name__=="__main__":

    # please do not modify the line below
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment

    low, high = min(env_info.vector_observations[0]), max(env_info.vector_observations[0])
    state_grid = create_uniform_grid(low, high, bins=30)
    state_grid = np.array(state_grid)

    action_size = 4
    state_size = 37

    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    # env.seed(0)
    # print('State shape: ', env.observation_space.shape)
    # print('Number of actions: ', env.action_space.n)

    scores = run_dql(env, agent, state_grid)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('episodexscore.png')
    plt.show()


