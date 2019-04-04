#!/usr/bin/env python
# coding: utf-8

# # Temporal-Difference Methods
# 
# In this notebook, you will write your own implementations of many Temporal-Difference (TD) methods.
# 
# While we have provided some starter code, you are welcome to erase these hints and write your code from scratch.
# 
# ---
# 
# ### Part 0: Explore CliffWalkingEnv
# 
# We begin by importing the necessary packages.

# In[3]:


import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt

import check_test
from plot_utils import plot_values


def generate_epsonde_from_Q(env, Q, epsilon=0):
    """Obtain an epsode from e-greedy police using Q-table"""
    epsodes = []

    # state = env.reset()
    state = env.observation_space.sample()
    i = 0
    while True:
        if state in Q:
            action = np.random.choice(np.arange(env.nA), p=get_probs(Q[state], epsilon=epsilon, nA=env.nA))
        else:
            action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        epsodes.append((state, action, reward))
        state = next_state
        i += 1
        if i > 10000:
            return epsodes
        if done:
            break
    return epsodes


def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    probs = np.ones(nA) * epsilon / (nA - 1)
    best_action = np.argmax(Q_s)
    probs[best_action] = 1 - epsilon

    return probs


def update_Q(Qsa, Qsa_next, reward, alpha, gamma):
    """ updates the action-value function estimate using the most recent time step """
    return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))


def epsilon_greedy_probs(Q_s, epsilon, nA, eps=None):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """

    policy_s = np.ones(nA) * epsilon / nA
    policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / nA)
    return policy_s


def sarsa(env, num_episodes, alpha, gamma=1.0):
    np.random.seed(928)
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    # initialize performance monitor
    # loop over episodes

    ## cutom code
    epsilon = 1
    num_episodes_concluded = 0
    ##
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{} - Epsilon: {}".format(i_episode, num_episodes, epsilon), end="")
            sys.stdout.flush()

            ## TODO: complete the function

        state = env.reset()
        epsilon = 1 / i_episode
        action = np.random.choice(np.arange(env.nA), p=epsilon_greedy_probs(Q[state], epsilon, env.nA))

        max_time_step_per_epoc = 300
        for i_step in range(max_time_step_per_epoc):

            next_state, reward, done, info = env.step(action)

            if done:
                num_episodes_concluded += 1
                break

            next_action = np.random.choice(np.arange(env.nA), p=get_probs(Q[next_state], epsilon, env.nA))

            Q[state][action] = Q[state][action] + alpha * (
                        reward + gamma * Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action

    print("\n\n{}/{} were completly finished".format(num_episodes_concluded, num_episodes))

    return Q


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


if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')

    # obtain the estimated optimal policy and corresponding action-value function
    Q_expsarsa = expected_sarsa(env, 10000, 0.3)

    # print the estimated optimal policy
    policy_expsarsa = np.array(
        [np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(
        4, 12)
    check_test.run_check('td_control_check', policy_expsarsa)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_expsarsa)

    # plot the estimated optimal state-value function

    # plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])
    # plot_values(V_opt)

