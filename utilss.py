import math

from utils.visualize.draw import Draw1, Draw2

def dfsLFT(nodes, i, LF,lowTime):

    if len(nodes[i].successor) == 0:

        LF[i] = lowTime
        return lowTime
    finishtime = 99999999

    for Orderid in nodes[i].successor:
        finishtime = min(finishtime, dfsLFT(nodes, Orderid, LF,lowTime))
    LF[i] = finishtime
    return finishtime - nodes[i].duration

def dfsEST(nodes, i, ES):

    if len(nodes[i].predecessor) == 0:
        ES[i] = 0
        return 0
    starttime = 0

    for Orderid in nodes[i].predecessor:
        starttime = max(starttime, dfsEST(nodes, Orderid, ES))
    ES[i] = starttime
    return starttime + nodes[i].duration

def find_top_10_percent_index(lst):
    num_elements = math.ceil(len(lst) * 0.1)
    sorted_lst = sorted(lst)
    top_10_percent = sorted_lst[-num_elements:]
    index_list = [lst.index(x) for x in top_10_percent]
    return index_list

def best_fitness_of_population(population,individuals_fitness):
    best_fitness = float('inf')
    best_individual = population[0]
    for i in range(len(population)):
        if individuals_fitness[i] < best_fitness:
            best_fitness = individuals_fitness[i]
            best_individual = population[i]
    return best_fitness,best_individual

def get_sorted_population(population):

    return  sorted(population, key=lambda x: x.fitness)

