import numpy as np
from random import randint
from sys import exit
from copy import deepcopy
from time import time

start_time = time()

"""
* Each chromosome is a 1D list of size (# of cols on chessboard). Each index
* will represent the row number of the queen for that column.
"""
class Chromosome:
    def __init__(self, length):
        self.length = length
        self.data = [0] * self.length
        self.conflicts = 0
        for i in range(self.length):
            self.data[i] = i

    def __repr__(self):
        representation = ""
        for i in range(self.length):
            representation += str(self.data[i]+1) + " "
        return representation

    def __lt__(self, chromosome2):
        return self.conflicts < chromosome2.conflicts

class Genetic:

    def __init__(self, length, num_chromosomes=100, initial_swaps=20, regularization=10):
        self.population = []
        self.length = length
        self.num_chromosomes = num_chromosomes
        self.regularization = regularization
        assert self.num_chromosomes % 4 == 0
        # Place queens randomly in all the chromosomes
        for i in range(self.num_chromosomes):
            chromosome = Chromosome(self.length)
            for j in range(initial_swaps):
                swap_index_1 = randint(0, self.length-1)
                swap_index_2 = randint(0, self.length-1)
                chromosome.data[swap_index_1], chromosome.data[swap_index_2] = chromosome.data[swap_index_2], chromosome.data[swap_index_1]
            self.fitness_function(chromosome)
            self.population.append(chromosome)

    def print_chromosomes(self):
        for i in range(self.num_chromosomes):
            print (self.population[i], "   ", self.population[i].conflicts)

    # The fitness function can be the number of queens that cannot be attacked
    # Only have to check the left side of each queen
    def fitness_function(self, chromosome):
        conflicts = 0
        board = np.zeros((self.length, self.length))
        
        # initialize a board for ease of checking and visualization
        for i in range(self.length):
            board[chromosome.data[i]][i] = 1

        for i in range(self.length):
            row = chromosome.data[i]
            col = i
            # Check Row
            for j in range(col-1, -1, -1):
                if board[row][j] == 1:
                    conflicts += 1
            
            # We dont have to check each col since our initialization is such
            # that each row has only one queen

            # Check diagonals

            j = row - 1
            k = col - 1
            while j >= 0 and k >= 0:
                if board[j][k] == 1:
                    conflicts += 1
                j -= 1
                k -= 1
            
            
            j = row + 1
            k = col - 1
            while k >= 0 and j < self.length:
                if board[j][k] == 1:
                    conflicts += 1
                j += 1
                k -= 1
        
        chromosome.conflicts = conflicts

    def sort_population(self):
        self.population.sort()

    def get_parent(self, parent, population):
        if parent is None:
            return randint(0, len(population)-1)
        else:
            new_parent = randint(0, len(population)-1)
            while new_parent == parent:
                new_parent = randint(0, len(population)-1)
            return new_parent

    def crossover_and_mutate(self):
        old_population = deepcopy(self.population)
        new_population = []

        for i in range(self.num_chromosomes//4):
            parent1 = old_population[self.get_parent(None, old_population)]
            parent2 = old_population[self.get_parent(parent1, old_population)]
            for j in range(4):
                split_pos = randint(0, self.length-2)
                child_data = parent1.data[:split_pos] + parent2.data[split_pos:]
                assert len(child_data) == self.length
                child = Chromosome(self.length)
                child.data = deepcopy(child_data)

                missing = []
                for k in range(self.length):
                    if k not in child.data:
                        missing.append(k)

                uniques = set({})
                for k in range(self.length):
                    if child.data[k] not in uniques:
                        uniques.add(child.data[k])
                    else:
                        removed = randint(0, len(missing) - 1)
                        child.data[k] = missing[removed]
                        del missing[removed]
                        uniques.add(child.data[k])

                self.fitness_function(child)
                new_population.append(child)
        if randint(1, 100) <= self.regularization:
            print ("Performing Regularization")
            new_population = new_population[:self.num_chromosomes//10]
            for i in range(self.num_chromosomes - self.num_chromosomes//10):
                chromosome = Chromosome(self.length)
                for j in range(20):
                    swap_index_1 = randint(0, self.length-1)
                    swap_index_2 = randint(0, self.length-1)
                    chromosome.data[swap_index_1], chromosome.data[swap_index_2] = chromosome.data[swap_index_2], chromosome.data[swap_index_1]
                self.fitness_function(chromosome)
                new_population.append(chromosome)
        assert len(new_population) == self.num_chromosomes
        return new_population
       
    def genetic(self, iterations):
        for i in range(iterations):
            genetic.sort_population()
            self.population = self.population[:self.num_chromosomes//4]
            
            # for j in range(5):
            #    print (self.population[j], "   ", self.population[j].conflicts)
            
            if self.population[0].conflicts == 0:
                global start_time
                end_time = time()
                print ("Solution Found at iteration", (i+1))
                self.print_solution(self.population[0])
                
                print ("Time Taken (s):", end_time-start_time)
                exit()
            
            self.population = self.crossover_and_mutate()
            # print ("-------------------------------------------------")
        
        print ("No Solution found")
        genetic.sort_population()
        for i in range(5):
            print ("Chromosome:", self.population[i], "; Fitness Value:", self.population[i].conflicts)

    def print_solution(self, chromosome):
        board = np.zeros((self.length, self.length))
        for i in range(self.length):
            board[chromosome.data[i]][i] = 1
        for i in range(self.length):
            for j in range(self.length):
                if board[i][j] == 1:
                    print ('Q', end=" ")
                else:
                    print ('.', end=" ")
            print()


        
if __name__ == '__main__':
    length = int(input())
    genetic = Genetic(length, num_chromosomes=1000)
    genetic.genetic(500)
    end_time = time()
    print ("Time Taken (s):", end_time-start_time)