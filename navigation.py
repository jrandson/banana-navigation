#!/usr/bin/env python
# coding: utf-8

import sys
import gym
import numpy as np
from collections import defaultdict, deque
from utils import create_uniform_grid, discretize
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt

import check_test
from plot_utils import plot_values


def test1(env):
    nA = env.nA
    Q_s = np.array([env.action_space.sample() for i in range(nA)])
    print("q_s", Q_s)
    epsilon = 0.3

    p = get_probs(Q_s, epsilon, nA)
    print(p)
    print(sum(p))
    p = epsilon_greedy_probs(Q_s, epsilon, nA)
    print(p)
    print(sum(p))


def test3(env):
    print(env.action_space)
    print(env.observation_space)

    # In[6]:

    env.observation_space.sample()
    state = env.reset()
    print(state)

    # In this mini-project, we will build towards finding the optimal policy for the CliffWalking environment.  The optimal state-value function is visualized below.  Please take the time now to make sure that you understand _why_ this is the optimal state-value function.

    # In[8]:

    # define the optimal state-value function
    V_opt = np.zeros((4, 12))
    V_opt[0:13][0] = -np.arange(3, 15)[::-1]
    V_opt[0:13][1] = -np.arange(3, 15)[::-1] + 1
    V_opt[0:13][2] = -np.arange(3, 15)[::-1] + 2
    V_opt[3][0] = -13

    # plot_values(V_opt)

    next_state, reward, done, info = env.step(env.action_space.sample())
    print(next_state)


def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    probs = np.ones(nA) * epsilon / (nA - 1)
    best_action = np.argmax(Q_s)
    probs[best_action] = 1 - epsilon

    return probs


def epsilon_greedy_probs(Q_s, epsilon, nA, eps=None):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """

    policy_s = np.ones(nA) * epsilon / nA
    policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / nA)
    return policy_s


# plot_values(V_opt)


def q_learning(env, num_episodes, alpha, gamma=1.0):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    # loop over episodes    
    epsilon = 0.6
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        # if i_episode % 100 == 0:
        print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
        sys.stdout.flush()
        ## TODO: complete the function              
        ####################################################        
        state = env.reset()
        epsilon = max(0.01, epsilon * 0.999)
        while True:
            # probs = get_probs(Q[state], epsilon, env.nA)
            probs = get_probs(Q[state], epsilon, env.nA)
            action = np.random.choice(np.arange(env.nA), p=probs)
            next_state, reward, done, info = env.step(action)
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            state = next_state
            if done:
                break
    return Q


def sarsa(env, num_episodes, state_grid, alpha, gamma=1.0):
    np.random.seed(928)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    nA = brain.vector_action_space_size
    Q = defaultdict(lambda: np.zeros(nA))

    epsilon = 0.7
    min_epsilon = 0.05
    decay_epsilon = 0.999

    num_episodes_concluded = 0

    scores = []
    max_avg_score = -np.inf


    for i_episode in range(1, num_episodes + 1):
        # monitor progress

        if i_episode % 100 == 0:
            print("\rEpisode {}/{} - Epsilon: {} Max avg score: {}".format(i_episode, num_episodes, epsilon, max_avg_score), end="")
            sys.stdout.flush()

        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        state = discretize(state, state_grid)

        epsilon = max(min_epsilon, decay_epsilon*epsilon)
        action = np.random.choice(np.arange(nA), p=epsilon_greedy_probs(Q[state], epsilon, nA))


        total_reward = 0

        while True:

            env_info = env.step(action)[brain_name]

            next_state = env_info.vector_observations[0]
            next_state = discretize(next_state, state_grid)
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            total_reward += reward

            if done:
                num_episodes_concluded += 1
                break

            next_action = np.random.choice(np.arange(nA), p=get_probs(Q[next_state], epsilon, nA))

            Q[state + (action,)] = Q[state + (action,)] + alpha * (
                        reward + gamma * Q[next_state + (next_action,)] - Q[state + (action,)])
            state = next_state
            action = next_action

        scores.append(total_reward)
        if len(scores) > 100:
            avg_score = np.mean(scores[-100:])
            if avg_score > max_avg_score:
                max_avg_score = avg_score

        if max_avg_score >= 13:
            print("The expect average score was bet. avg score: {}".format(max_avg_score))
            break


    print("\n\n{}/{} were completly finished".format(num_episodes_concluded, num_episodes))

    return Q, scores


def expected_sarsa(env, num_episodes, alpha, gamma=1.0):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    prob_police = defaultdict(lambda: np.zeros(env.nA))
    # loop over episodes
    epsilon = 0
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        ## TODO: complete the function
        state = env.reset()
        while True:
            probs = get_probs(Q[state], epsilon, env.nA)
            action = np.random.choice(np.arange(env.nA), p=probs)
            next_state, reward, done, info = env.step(action)

            probs = get_probs(Q[next_state], epsilon, env.nA)
            Q[state][action] = Q[state][action] + alpha * (
                        reward + gamma * np.dot(probs, Q[next_state]) - Q[state][action])
            state = next_state

            if done:
                break

    return Q


if __name__ == "__main__":
    # please do not modify the line below
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment

    low, high = min(env_info.vector_observations[0]), max(env_info.vector_observations[0])
    print("Low: {} High: {}".format(low, high))
    state_grid = create_uniform_grid(low, high, bins=30)
    state_grid = np.array(state_grid)

    # q_agent = QLearningAgent(env, state_grid, brain_name, alpha=0.1, gamma=0.99,
    #                          epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.05, seed=505)

    Q_table, scores = sarsa(env, 5000, state_grid, alpha=0.3, gamma=0.7)

    # print the estimated optimal policy
    # policy_expsarsa = np.array(
    #     [np.argmax(Q_table[key]) if key in Q_table else -1 for key in np.arange(48)]).reshape(
    #     4, 12)
    # check_test.run_check('td_control_check', policy_expsarsa)
    # print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    # print(policy_expsarsa)

    plt.figure()
    plt.plot(scores)
    plt.show()

    # plot the estimated optimal state-value function

    # plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])
    # plot_values(V_opt)

