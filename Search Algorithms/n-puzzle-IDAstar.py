import numpy as np
from copy import deepcopy
from sys import exit
from math import sqrt, ceil
from time import time

start_time = time()

class Node:
    
    def __init__(self, state, x, y):
        self.state = state
        self.x = x
        self.y = y
        self.n = len(state)
        self.parent = 0
        self.move_index = 0
        self.f = 0
        self.g = 0

    def __repr__(self):
        s = ""
        for i in range(self.n):
            for j in range(self.n):
                s += str(self.state[i][j]) + " "
            s += "\n"
        s += "\n"
        return s

class IDAstar:

    def __init__(self, initial_state, final_state, n):
        self.initial_state = initial_state
        self.final_state = final_state
        self.n = n
        self.moves = [(1,0), (-1,0), (0,1), (0,-1)]
        self.threshold = 0
        self.explored = set({})
        self.stack = []
        self.newthreshold = np.inf

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

    def generate(self, node, x, y):

        """
        * Check for an invalid state
        """
        if node.x + x < 0 or node.x + x >= self.n or node.y + y < 0 or node.y + y >= self.n:
            return -1
        
        newnode = Node(deepcopy(node.state), node.x, node.y)
        newnode.state[newnode.x][newnode.y], newnode.state[newnode.x + x][newnode.y + y] = newnode.state[newnode.x + x][newnode.y + y], newnode.state[newnode.x][newnode.y]       

        
        # print ("OLD")
        # print (node)
        # print ("NEW")
        # print (newnode)
        # print (self.is_explored(newnode.__repr__()))
        # print (node.move_index)
        # print ("---------------------------------")

        if self.is_explored(newnode.__repr__()):
            return -1
        newnode.g = node.g + 1
        newnode.f = newnode.g + self.heuristic(newnode.state)
        # print ("newnode", newnode.f)
        if newnode.f > self.threshold:
            self.newthreshold = min(self.newthreshold, newnode.f)
            return -1

        newnode.parent = node
        newnode.x += x
        newnode.y += y

        self.explored.add(newnode.__repr__())
        return newnode

    def idastar(self):
        initial_node = Node(self.initial_state, np.where(self.initial_state == -1)[0][0], np.where(self.initial_state == -1)[1][0])
        self.threshold = self.heuristic(initial_node.state)
        
        gl = 0 # Limit to how many times we run a new threshold
        while gl < 1000:
            self.explored = set({}) # Re-initialize the explored states
            self.explored.add(initial_node.__repr__())
            curnode = deepcopy(initial_node)
            node_remains = True
            while node_remains:
                if self.is_goal(curnode.state):
                    global start_time
                    end_time = time()
                    self.print_results(curnode)
                    print ("Time Taken (s):", end_time-start_time)
                    exit()
                while curnode.move_index < 4:
                    nextnode = self.generate(curnode, self.moves[curnode.move_index][0], self.moves[curnode.move_index][1])
                    curnode.move_index += 1

                    if nextnode != -1:
                        self.stack.append(curnode)
                        curnode = Node(deepcopy(nextnode.state), nextnode.x, nextnode.y)
                        curnode.parent = nextnode.parent
                        curnode.g = nextnode.g
                
                while curnode.move_index == 4 and len(self.stack) > 0:
                    curnode = self.stack.pop()
                if curnode.move_index == 4:
                    node_remains = False
            #print (self.threshold, self.newthreshold)
            self.threshold = self.newthreshold
            self.newthreshold = np.inf
            gl += 1
            
if __name__ == '__main__':
    # n = 4
    # initial_state = np.asarray([6,2,3,4,1,5,8,-1,9,10,7,12,13,14,11,15]).reshape((4, 4))
    # final_state = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,-1]).reshape((4, 4))
    # n = 5
    # initial_state = np.asarray([1,2,4,9,5,6,8,3,14,10,11,7,-1,12,15,16,17,13,18,20,21,22,23,19,24]).reshape((5, 5))
    # final_state = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,-1]).reshape((5,5))
    # n = 3
    # initial_state = np.asarray([[-1,3,8], [4,1,7], [2,6,5]])
    # final_state = np.asarray([[-1,1,2], [3,4,5], [6,7,8]])
    n = 3
    initial_state = np.asarray([[1,2,3], [8,-1,4], [7,6,5]])
    final_state = np.asarray([[1,2,3], [7,8,4], [-1,6,5]])
    IDAstarop = IDAstar(initial_state, final_state, n)
    IDAstarop.idastar()