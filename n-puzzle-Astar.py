from queue import PriorityQueue
from copy import deepcopy
import numpy as np
from math import sqrt
from time import time
from sys import exit

start_time = time()

class Node:
    
    def __init__(self, state, x, y):
        self.state = state
        self.x = x
        self.y = y
        self.n = len(state)
        self.parent = 0
        self.f = 0
        self.g = 0

    def __lt__(self, node):
        return self.f < node.f

    def __repr__(self):
        s = ""
        for i in range(self.n):
            for j in range(self.n):
                s += str(self.state[i][j]) + " "
            s += "\n"
        s += "\n"
        return s
        
class Astar:

    def __init__(self, initial_state, final_state, n):
        self.pqueue = PriorityQueue()
        self.initial_state = initial_state
        self.final_state = final_state
        self.n = n
        self.explored = set({})
        self.moves = [(1,0), (-1,0), (0,1), (0,-1)]
        

    def print_results(self, node):
        stack = []
        path_length = 0
        while not np.array_equal(node.state, self.initial_state):
            stack.append(node)
            node = node.parent
            path_length += 1
        stack.append(node)

        
        while len(stack) > 0:
            print (stack.pop())

        print ("Path Length:", path_length)
        

    def is_goal(self, state):
        for i in range(self.n):
            for j in range(self.n):
                if state[i][j] != self.final_state[i][j]:
                    return False
        return True

    def is_explored(self, state):
        return state in self.explored

    def heuristic(self, state):
        h = 0.0
        for i in range(self.n):
            for j in range(self.n):
                x, y = i, j
                x1, y1 = np.where(self.final_state == state[i][j])[0][0], np.where(self.final_state == state[i][j])[1][0]
                h += sqrt(pow(x1-x, 2) + pow(y1-y, 2))
        return h

    def generate(self, node, x, y):
        
        """
        * Check for an invalid state
        """
        if node.x + x < 0 or node.x + x >= self.n or node.y + y < 0 or node.y + y >= self.n:
            return
        
        newnode = deepcopy(node)
        newnode.state[newnode.x][newnode.y], newnode.state[newnode.x + x][newnode.y + y] = newnode.state[newnode.x + x][newnode.y + y], newnode.state[newnode.x][newnode.y]

        """
        print ("OLD")
        print (node)
        print ("NEW")
        print (newnode)
        print (self.is_explored(newnode.__repr__()))
        print ("---------------------------------")
        """

        if self.is_explored(newnode.__repr__()):
            return

        newnode.parent = node
        newnode.x += x
        newnode.y += y
        newnode.g = node.g + 1
        newnode.f = newnode.g + self.heuristic(newnode.state)

        self.explored.add(newnode.__repr__())
        self.pqueue.put(newnode)
        

    def expand(self, node):
        
        if self.is_goal(node.state):
            global start_time
            end_time = time()
            self.print_results(node)
            print ("Time Taken (s):", end_time-start_time)
            exit()
        i = 0
        while i < 4:
            self.generate(node, self.moves[i][0], self.moves[i][1])
            i+=1


    def astar(self):
        
        initial_node = Node(self.initial_state, np.where(self.initial_state == -1)[0][0], np.where(self.initial_state == -1)[1][0])
        self.explored.add(initial_node.__repr__())
        self.pqueue.put(initial_node)
        while not self.pqueue.empty():
            current_node = self.pqueue.get()
            self.expand(current_node)


if __name__ == '__main__':
    
    n = 4
    initial_state = np.asarray([6,2,3,4,1,5,8,-1,9,10,7,12,13,14,11,15]).reshape((4, 4))
    final_state = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,-1]).reshape((4, 4))
    # n = 5
    # initial_state = np.asarray([1,2,4,9,5,6,8,3,14,10,11,7,-1,12,15,16,17,13,18,20,21,22,23,19,24]).reshape((5, 5))
    # final_state = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,-1]).reshape((5,5))
    # n = 3
    # initial_state = np.asarray([[-1,3,8], [4,1,7], [2,6,5]])
    # final_state = np.asarray([[-1,1,2], [3,4,5], [6,7,8]])
    # n = 3
    # initial_state = np.asarray([[1,2,3], [8,-1,4], [7,6,5]])
    # final_state = np.asarray([[1,2,3], [7,8,4], [-1,6,5]])
    Astarop = Astar(initial_state, final_state, n)
    Astarop.astar()
    
