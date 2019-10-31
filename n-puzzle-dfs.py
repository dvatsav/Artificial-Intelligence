import queue as queue_ds
import numpy as np
from copy import deepcopy
from sys import exit
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

    def __repr__(self):
        s = ""
        for i in range(self.n):
            for j in range(self.n):
                s += str(self.state[i][j]) + " "
            s += "\n"
        s += "\n"
        return s

class DFS:
    def __init__(self, initial_state, final_state, n):
        self.stack = []
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

        newnode.parent = node
        newnode.x += x
        newnode.y += y

        if self.is_goal(newnode.state):
            global start_time
            end_time = time()
            self.print_results(newnode)
            print ("Time Taken:", end_time-start_time)
            exit()

        self.explored.add(newnode.__repr__())
        return newnode

    def dfs(self):
        initial_node = Node(self.initial_state, np.where(self.initial_state == -1)[0][0], np.where(self.initial_state == -1)[1][0])
        self.explored.add(initial_node.__repr__())
        curnode = deepcopy(initial_node)
        node_remains = True
        while node_remains:
            while curnode.move_index < 4:
                nextnode = self.generate(curnode, self.moves[curnode.move_index][0], self.moves[curnode.move_index][1])
                curnode.move_index += 1

                if nextnode != -1:
                    self.stack.append(curnode)
                    curnode = Node(deepcopy(nextnode.state), nextnode.x, nextnode.y)
                    curnode.parent = nextnode.parent
            
            while curnode.move_index == 4 and len(self.stack) > 0:
                curnode = self.stack.pop()
            if curnode.move_index == 4:
                node_remains = False
            
            
if __name__ == '__main__':
    # n = 3
    # initial_state = np.asarray([[1,2,3], [8,-1,4], [7,6,5]])
    # final_state = np.asarray([[1,2,3], [7,8,4], [-1,6,5]])
    n = 4
    initial_state = np.asarray([1,2,3,-1,5,6,7,4,9,10,11,8,13,14,15,12]).reshape((4, 4))
    final_state = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,-1]).reshape((4, 4))
    DFSop = DFS(initial_state, final_state, n)
    DFSop.dfs()