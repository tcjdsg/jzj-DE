import math


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
