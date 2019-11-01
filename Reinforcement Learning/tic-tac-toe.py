"""
* Multi Agent Tic Tac Toe using Q-Learning
* Deepak Srivatsav, IIIT Delhi
"""

import numpy as np
from random import random as rand, randint, choice
import sys
import pickle
import matplotlib.pyplot as plt
import time

epsilon = 0.1
learning_rate = 0.1
gamma = 0.9
win_reward = 1000
lose_reward = -5000
tie_reward = 100
step_reward = -5
num_games = 50000

"""
* Function for getting the action with maximum Q value
"""
def get_max_action(Q, state):
    max_actions = []
    max_act_value = -np.inf
    for act in Q[state]:
        if Q[state][act] > max_act_value:
            max_actions = [act]
            max_act_value = Q[state][act]
        elif Q[state][act] == max_act_value:
            max_actions.append(act)
    return choice(max_actions)

"""
* Get an action based on the current state, follows e-greedy policy
"""
def get_action(Q, state):
    global epsilon
    
    def get_random_action(Q, state):
        available_actions = []
        for act in Q[state]:
            available_actions.append(act)
        return choice(available_actions)
    
    if rand() < epsilon:
        return get_random_action(Q, state)
    else:
        return get_max_action(Q, state)

def reset_env():
    return (0, 0, 0, 0, 0, 0, 0, 0, 0)

"""
* Indices where the player can play follow 0 based indexing, from left to right on the board
"""
def init_state(Q, state):
    Q[state] = {}
    for ind, pos in enumerate(state):
        if pos == 0:
            Q[state][ind] = 0
    return Q

"""
* Take an action and calculate rewards for each player along with next state
"""
def step(action, state, player):
    
    def check_state(next_state):
        st = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                st[i][j] = next_state[i*3 + j]

        # print ("State st:\n", st)
        
        # Check all rows
        for i in range(3):
            if np.sum(st[i]) == 3:
                return win_reward, lose_reward, True
            if np.sum(st[i]) == -3:
                return lose_reward, win_reward, True

        st = st.transpose()
        # Check all columns
        for i in range(3):
            if np.sum(st[i]) == 3:
                return  win_reward, lose_reward, True
            if np.sum(st[i]) == -3:
                return lose_reward, win_reward, True

        # Check Diagonals
        sum_major_diag = 0
        sum_minor_diag = 0
        for i in range(3):
            sum_major_diag += st[i][i]
            sum_minor_diag += st[2-i][i]
        
        if sum_major_diag == 3 or sum_minor_diag == 3:
            return win_reward, lose_reward, True
        elif sum_major_diag == -3 or sum_minor_diag == -3:
            return lose_reward, win_reward, True
        
        if 0 not in next_state:
            return tie_reward, tie_reward, True

        return step_reward, step_reward, False

    next_state = state[:action] + (player,) + state[action+1:]
    reward_p1, reward_p2, done = check_state(next_state)
    return next_state, reward_p1, reward_p2, done

"""
* Print the Current State to stdout
"""
def render(state, reward_p1=0, reward_p2=0):
    print ("\n-------------          -------------")
    game = [0] * 9
    for i in range(9):
        if state[i] == 1:
            game[i] = "X"
        elif state[i] == -1:
            game[i] = "O"
        else:
            game[i] = " "
    for i in range(3):
        print ("| ", end="")
        for j in range(3):
            print (game[i*3+j], end=" | ")
        if i == 0:
            print ("         | 0 | 1 | 2 |", end="")
        elif i == 1:
            print ("         | 3 | 4 | 5 |", end="")
        else:
            print ("         | 6 | 7 | 8 |", end="")
        print ("\n-------------          -------------")
    
    print ("\nPlayer 1 Reward:", reward_p1, "; Player 2 Reward:", reward_p2)
    print ("\n\n")


"""
* Train Agents
"""
def train():
    
    global num_games
    global gamma
    global learning_rate

    Q1 = {}
    Q2 = {}

    td_error_p1 = []
    td_error_p2 = []
    cumulative_reward_p1 = []
    cumulative_reward_p2 = []

    for _ in range(num_games):
        state = reset_env()
        p1prevstate = -1
        p1prevaction = -1
        p2prevstate = -1
        p2prevaction = -1
        sumreward_p1 = 0
        sumreward_p2 = 0
        while True:
            
            player = 1
            if state not in Q1:
                Q1 = init_state(Q1, state)
            action = get_action(Q1, state)
            p1prevstate = state
            p1prevaction = action
            next_state, reward_p1, reward_p2, done = step(action, state, player)
            sumreward_p1 += reward_p1
            sumreward_p2 += reward_p2
            
            if next_state not in Q2:
                Q2 = init_state(Q2, next_state)
            
            if p2prevaction != -1 and p2prevstate != -1 and not done:
                Q2[p2prevstate][p2prevaction] += learning_rate * (reward_p2 + gamma * Q2[next_state][get_max_action(Q2, next_state)] - Q2[p2prevstate][p2prevaction])
                td_error_p2.append(np.abs(reward_p2 + gamma * Q2[next_state][get_max_action(Q2, next_state)] - Q2[p2prevstate][p2prevaction]))
            if done:
                Q1[p1prevstate][p1prevaction] += learning_rate * (reward_p1 - Q1[p1prevstate][p1prevaction])
                Q2[p2prevstate][p2prevaction] += learning_rate * (reward_p2 - Q2[p2prevstate][p2prevaction])
                td_error_p1.append(np.abs(reward_p1 - Q1[p1prevstate][p1prevaction]))
                td_error_p2.append(np.abs(reward_p2 - Q2[p2prevstate][p2prevaction]))
                break

            player = -1

            state = next_state
            action = get_action(Q2, state)
            p2prevstate = state
            p2prevaction = action

            
            next_state, reward_p1, reward_p2, done = step(action, state, player)
            sumreward_p1 += reward_p1
            sumreward_p2 += reward_p2
            
            if next_state not in Q1:
                Q1 = init_state(Q1, next_state)

            if p1prevaction != -1 and p1prevstate != -1 and not done:
                Q1[p1prevstate][p1prevaction] += learning_rate * (reward_p1 + gamma * Q1[next_state][get_max_action(Q1, next_state)] - Q1[p1prevstate][p1prevaction])
                td_error_p1.append(np.abs(reward_p1 + gamma * Q1[next_state][get_max_action(Q1, next_state)] - Q1[p1prevstate][p1prevaction]))
            if done:
                Q2[p2prevstate][p2prevaction] += learning_rate * (reward_p2 - Q2[p2prevstate][p2prevaction])
                Q1[p1prevstate][p1prevaction] += learning_rate * (reward_p1 - Q1[p1prevstate][p1prevaction])
                td_error_p1.append(np.abs(reward_p1 - Q1[p1prevstate][p1prevaction]))
                td_error_p2.append(np.abs(reward_p2 - Q2[p2prevstate][p2prevaction]))
                break
 
            state = next_state
        cumulative_reward_p1.append(sumreward_p1)
        cumulative_reward_p2.append(sumreward_p2)
        if (_ % 10000 == 0):
            print ("[*] %d/%d games complete" % (_ + 1, num_games))
    return Q1, Q2, td_error_p1, td_error_p2, cumulative_reward_p1, cumulative_reward_p2

"""
* Play Against a Trained Agent
"""

def play(Q1, Q2):
    print ("Do you wish to be X or O? (X starts first): ", end="")
    opt = input().lower()
    Q = {}
    if opt == "x":
        Q = Q2
    elif opt == "o":
        Q = Q1
    else:
        print ("Invalid Choice")
        sys.exit()
    
    if opt == "o":
        state = reset_env()
        render(state)
        while True:
            player = 1
            action = get_max_action(Q, state)
            next_state, reward_p1, reward_p2, done = step(action, state, player)
            render(next_state, reward_p1, reward_p2)
            if done:
                break
            
            state = next_state
            player = -1
            max_turns = 5
            for _ in range(max_turns):
                print ("Enter a valid move: ", end="")
                try:
                    action = int(input())
                except:
                    print ("Invalid Action")
                if state[action] == 0:
                    break
                else:
                    print ("Invalid Action")
            next_state, reward_p1, reward_p2, done = step(action, state, player)
            render(next_state, reward_p1, reward_p2)
            if done:
                break
            state = next_state
    else:
        state = reset_env()
        render(state)
        while True:
            player = 1
            max_turns = 5
            for _ in range(max_turns):
                print ("Enter a valid move: ", end="")
                try:
                    action = int(input())
                except:
                    print ("Invalid Action")
                if state[action] == 0:
                    break
                else:
                    print ("Invalid Action")
            next_state, reward_p1, reward_p2, done = step(action, state, player)
            render(next_state, reward_p1, reward_p2)
            if done:
                break
            state = next_state
            player = -1
            action = get_max_action(Q, state)
            next_state, reward_p1, reward_p2, done = step(action, state, player)
            render(next_state, reward_p1, reward_p2)
            if done:
                break
            
            state = next_state
            

if __name__ == '__main__':
    print ("Train or Play? (T/P): ", end="")
    opt = input().lower()
    if opt == "t":
        print ("[*] Started Training")
        start_time = time.time()
        Q1, Q2, td_error_p1, td_error_p2, cumulative_reward_p1, cumulative_reward_p2 = train()
        end_time = time.time()
        print ("[*] Train Time: %fs" %(end_time - start_time))
        print ("[*] Training Complete, Saving Q values")
        pickle_out = open("Player1Q.pickle","wb")
        pickle.dump(Q1, pickle_out)
        pickle_out.close()
        pickle_out = open("Player2Q.pickle","wb")
        pickle.dump(Q2, pickle_out)
        pickle_out.close()
        print ("[*] Saved Q values")
        plt.figure(1)
        plt.plot(td_error_p1)
        plt.xlabel("Iterations")
        plt.ylabel("TD Error")
        plt.title("TD Error vs Iterations - P1")
        plt.figure(2)
        plt.plot(td_error_p2)
        plt.xlabel("Iterations")
        plt.ylabel("TD Error")
        plt.title("TD Error vs Iterations - P2")
        plt.figure(3)
        plt.plot(cumulative_reward_p1)
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward per Episode - P1")
        plt.figure(4)
        plt.plot(cumulative_reward_p2)
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward per Episode - P2")
        plt.show()
    elif opt == "p":
        try:
            pickle_in = open("Player1Q.pickle","rb")
            Q1 = pickle.load(pickle_in)
            pickle_in = open("Player2Q.pickle","rb")
            Q2 = pickle.load(pickle_in)
            play(Q1, Q2)
        except:
            print ("You have to train before playing")
    else:
        print ("Invalid Choice")

