import numpy as np
from random import random as rand, randint, choice, seed
import networkx as nx
import matplotlib.pyplot as plt
import time

seed(1)

class Graph():
    def __init__(self, n, dist_matrix):
        self.n = n
        self.dist_matrix = dist_matrix
        self.pheromones = np.zeros((self.n, self.n))
        self.pheromones += 1/self.n**2
        

class Ant():
    def __init__(self, start_city, g):
        self.start_city = start_city
        self.cur_city = self.start_city
        self.tour = [self.cur_city]
        self.tour_cost = 0
        self.allowed_cities = [i for i in range(g.n) if g.dist_matrix[start_city][i] >= 0]
        self.allowed_cities.remove(start_city)


class ACO():
    def __init__(self, graph, alpha, beta, rho, num_generations):
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.num_generations = num_generations
        self.ants = [Ant(i, self.graph) for i in range(self.graph.n)]
        self.num_ants = self.graph.n

    def get_next_city(self, ant):

        if len(ant.allowed_cities) == 0:
            return -1

        probabilities = np.zeros(len(ant.allowed_cities), dtype=np.float)
        total_sum_weight = 0
        
        for ind, city in enumerate(ant.allowed_cities):
            probabilities[ind] = self.graph.pheromones[ant.cur_city][city]**self.alpha * (1/self.graph.dist_matrix[ant.cur_city][city])**self.beta
            total_sum_weight += probabilities[ind]
        if total_sum_weight != 0:
            probabilities /= total_sum_weight

        next_city = ant.allowed_cities[np.argmax(probabilities)]
        ant.allowed_cities.remove(next_city)
        return next_city

    def update_pheromones(self):
        traveling_ants = np.zeros((self.graph.n, self.graph.n), dtype=np.float)
        for ant in self.ants:
            tour = ant.tour
            for i in range(len(tour)-1):
                traveling_ants[tour[i]][tour[i+1]] += 1/self.graph.dist_matrix[tour[i]][tour[i+1]]
        traveling_ants *= self.rho
        
        self.graph.pheromones *= (1 - self.rho)
        self.graph.pheromones += traveling_ants

    def run(self):
        best_tour_cost = np.inf
        best_tour = []
        for i in range(self.num_generations):
            self.ants = [Ant(i, self.graph) for i in range(self.graph.n)]
            for ind, ant in enumerate(self.ants):
                cur_city = ant.cur_city
                for j in range(self.graph.n - 1):
                    
                    next_city = self.get_next_city(ant)
                    # print ("Cur Ant:", ind, "Cur City:", cur_city, "Next City", next_city)
                    if next_city == -1:
                        break
                    ant.tour.append(next_city)
                    ant.tour_cost += self.graph.dist_matrix[cur_city][next_city]
                    cur_city = next_city
                ant.tour_cost += self.graph.dist_matrix[cur_city][ant.start_city]
                ant.tour.append(ant.start_city)
            ant_tour_lengths = np.array(list(map(lambda x: x.tour_cost, self.ants)))
            best_gen_cost = np.min(ant_tour_lengths)
            if best_gen_cost < best_tour_cost:
                best_tour_cost = best_gen_cost
                best_tour = self.ants[np.argmin(ant_tour_lengths)].tour
        
            self.update_pheromones()
        return best_tour_cost, best_tour
        

def main():
    def gen_dist_matrix(n):
        dist_mat = np.zeros((n, n))
        dist_mat += 0
        for i in range(n):
            for j in range(i+1, n):
                dist_mat[i][j] = randint(1, 100)
                dist_mat[j][i] = dist_mat[i][j]

        return dist_mat

    def plot(dist_mat, n, tour = []):
        g = nx.Graph()
        tour_edges = []
        if len(tour) > 0:
            for i in range(len(tour) - 1):
                tour_edges.append((tour[i], tour[i+1]))
        for i in range(n):
            g.add_node(i)
            for j in range(i+1, n):
                if (i, j) in tour_edges or (j, i) in tour_edges:
                    g.add_edge(i, j, color='b', weight=dist_mat[i][j])
                else:
                    g.add_edge(i, j, color='r', weight=dist_mat[i][j])
        labels = nx.get_edge_attributes(g,'weight')
        pos=nx.spring_layout(g)
        colors = [g[u][v]['color'] for u,v in g.edges()]
        nx.draw(g, pos, edge_color=colors, with_labels=True)
        nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
        plt.show()


    n = 20
    alpha = 2
    beta = 0
    num_generations = 100
    rho = 0.5 # pheromone decay factor
    dist_mat = gen_dist_matrix(n)
    plt.clf()
    plot(dist_mat, n)
    graph = Graph(n=n, dist_matrix=dist_mat)
    start_time = time.time()
    aco = ACO(graph, alpha, beta, rho, num_generations)
    best_tour_cost, best_tour = aco.run()
    end_time = time.time()
    plt.clf()
    plt.title("Best Tour Cost: %d, Best Tour: " %best_tour_cost + str(best_tour))
    plot(dist_mat, n, best_tour)

    print ("Best Tour Cost: %d" %best_tour_cost)
    print ("Best Tour: ", best_tour)
    print ("Time Taken: %fs" %(end_time-start_time))
    print ("Distance Matrix")
    print (dist_mat)


if __name__ == '__main__':
    main()