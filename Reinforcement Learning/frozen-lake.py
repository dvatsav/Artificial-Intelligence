import gym
import numpy as np
from random import random as rand, randint, choice
import matplotlib.pyplot as plt
import time

num_episodes = 10000
max_episode_len = 10000
alpha = 0.9
gamma = 0.9
epsilon = 0.1
actions = {0:'left', 1:'down', 2:'right', 3:'up'}


"""
* Observation 1 - Using a greedy policy leads to the agent being stuck on 0,0 - always taking action left. Hence, an e-greedy policy will be used
"""

def get_action(env, Q, state):
    if rand() < epsilon:
        action = randint(0, env.action_space.n-1)
    else:
        actions = []
        for act, val in enumerate(Q[state, :]):
            if val == np.max(Q[state, :]):
                actions.append(act)
        action = choice(actions)

    return action

def render(env, action, reward, state, next_state):
    desc = env.desc
    for i in range(len(desc)):
        for j in range(len(desc[0])):
            print (desc[i][j].decode('utf-8'), end=" ")
        print ()
    print ("Action:", actions[action], ", Reward:", reward, ", Current State:", state, ", Next State:", next_state)
    print ()

def train(env):
    global epsilon
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    td_error = []
    cumulative_reward = []
    for _ in range(num_episodes):
        state = env.reset()
        sumreward = 0
        while True:
            action = get_action(env, Q, state)
            next_state, reward, done, info = env.step(action)
            # render(env, action, reward, state, next_state)
            next_action = np.argmax(Q[next_state, :])
            Q[state, action] += alpha * (reward + gamma*Q[next_state, next_action] - Q[state, action])
            td_error.append(reward + gamma*Q[next_state, next_action] - Q[state, action])
            sumreward += reward
            if done:
                break
            state = next_state
        cumulative_reward.append(sumreward)
    return Q, td_error, cumulative_reward

def play(env, Q):

    state = env.reset()
    for _ in range(max_episode_len):
        action = np.argmax(Q[state, :])
        next_state, reward, done, info = env.step(action)
        render(env, action, reward, state, next_state)
        state = next_state
        if done:
            print ("You Win!!")
            break
    
def main():
    env = gym.make('FrozenLake-v0', is_slippery=False)
    start_time = time.time()
    Q, td_error, cumulative_reward = train(env)
    end_time = time.time()
    print ("\nDone Training------------------------")
    print ("Training Time: %fs" %(end_time - start_time))
    print ("\nLet's Play!")
    play(env, Q)
    plt.figure(1)
    plt.plot(td_error)
    plt.xlabel("Iterations")
    plt.ylabel("Temporal Difference Error per iteration")
    plt.figure(2)
    plt.plot(cumulative_reward)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward per Episode")
    plt.show()
    



if __name__ == '__main__':
    main()