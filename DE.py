import copy
import math
from random import uniform

import numpy as np

from calFitness import Fitness
from initm import InitM


def run_diferential_evolution_for_JZJ(activities,max_generations,  NP, F, CR):
    count = 0
    D = len(list(activities.keys()))
    generation_of_best_fitness = 0

    population = [[0 for _ in range(D)] for _ in range(NP)]
    individuals_fitness = [0.0]*NP
    lower_bound = [-1]*D
    upper_bound = [1]*D
    best_individual = [0]*D
    best_global_fitness = float('inf')
    this_population_best_fitness = 0.0

    initialize_individuals_randomly(population, activities,lower_bound, upper_bound, individuals_fitness, NP, D)

    while count < max_generations:
        population,individuals_fitness = DE_mutate_recombine_evaluate_and_select(population, individuals_fitness, NP, D, F, CR,'l')

        this_population_best_fitness,best_individual = best_fitness_of_population(individuals_fitness, NP)
        if this_population_best_fitness < best_global_fitness:
            best_global_fitness = this_population_best_fitness
            generation_of_best_fitness = count
        count += 1

    print("{} {} {} {} {} {} {}".format(D, NP, F, CR, max_generations, best_global_fitness, generation_of_best_fitness))

    del individuals_fitness[:]
    del lower_bound[:]
    del upper_bound[:]

from pyDOE import lhs


def initialize_individuals_randomly(population, activities,lower_bound, upper_bound, individuals_fitness, NP, D):
    #uniform(lower_bound[j], upper_bound[j])

    for j in range(D):
        # 拉丁超立方采样
        xx = lower_bound[j] + (upper_bound[j] - lower_bound[j]) * lhs(1, NP)
        for i in range(NP):
            population[i][j] = xx[i]
    for i in range(NP):
        individuals_fitness[i] = DE_evaluate(population[i], activities,'l')

def DE_evaluate(individual, activities,LR):
    evaluation_up_to_date = 1
    fitness = Fitness(individual, copy.deepcopy(activities),LR)
    return fitness
import random


def DE_mutate_recombine_evaluate_and_select(population, individuals_fitness, NP, D, F, CR,LR):
    # trial populations is used as nex population in select
    trial_population = [[0 for _ in range(D)] for _ in range(NP)]
    trials_fitness = [0.0] *NP

    for i in range(NP):
        DE_mutate_and_recombine(population, i, trial_population[i], NP, D, F, CR)
        trials_fitness[i] = DE_evaluate(trial_population[i], D,LR)

    for i in range(NP):
        individuals_fitness[i],trial_population[i] = DE_select(population[i], individuals_fitness[i], trial_population[i], trials_fitness[i])

    population = copy.deepcopy(trial_population)

    del trial_population[:]
    del trials_fitness[:]
    return population,individuals_fitness


def DE_select(individual, fitness_of_individual, trial_vector, fitness_of_trial_vector):
    if fitness_of_trial_vector <= fitness_of_individual:
        fitness_of_individual = fitness_of_trial_vector
    else:
        trial_vector = copy.deepcopy(individual)
    return fitness_of_individual,trial_vector

# DE/rand/1
def DE_rand_1_mutate_and_recombine(population, individual_index, trial_vector, NP, D, F, CR):
    # Randomly pick 3 vectors, all different from individual_index
    ids = []
    for id in range(0,NP):
        if id!=individual_index:
            ids.append(id)

    pp = random.sample(ids,3)
    a = pp[0]
    b=pp[1]
    c=pp[2]
    # Randomly pick an index for forced evolution change
    k = int(random.uniform(0, D))

    # Load D parameters into trial_vector[]
    for j in range(D):
        # Perform NP-1 binomial trials.
        if random.uniform(0.0, 1.0) < CR or j == k:
            # Source for trial_vector[j] is a random vector plus weighted differential
            trial_vector[j] = population[c][j] + F * (population[a][j] - population[b][j])
        else:
            # or trial_vector parameter comes from population[individual_index][j] itself.
            trial_vector[j] = population[individual_index][j]

def DE_best_1_mutate_and_recombine(population,best_individual, individual_index, trial_vector, NP, D, F, CR):
    ids = []
    for id in range(0, NP):
        if id!=individual_index:
            ids.append(id)

    pp = random.sample(ids,2)
    a = pp[0]
    b = pp[1]
    k = int(random.uniform(0, D))
  #  best_fitness ,best_individual = best_fitness_of_population(population,individuals_fitness)
    # Load D parameters into trial_vector[]
    for j in range(D):
        # Perform NP-1 binomial trials.
        if random.uniform(0.0, 1.0) < CR or j == k:
            # Source for trial_vector[j] is a random vector plus weighted differential
            trial_vector[j] = best_individual[j] + F * (population[a][j] - population[b][j])
        else:
            # or trial_vector parameter comes from population[individual_index][j] itself.
            trial_vector[j] = population[individual_index][j]


def DE_current_to_pbest_1_mutate_and_recombine(population, top_10_individuals,individual_index, trial_vector, NP, D, F, CR):
    ids = []
    for id in range(0, NP):
        if id != individual_index:
            ids.append(id)

    pp = random.sample(ids, 2)
    a = pp[0]
    b = pp[1]
    k = int(random.uniform(0, D))
    best_individual = top_10_individuals[random.randint(0, len(top_10_individuals))]
    #  best_fitness ,best_individual = best_fitness_of_population(population,individuals_fitness)
    # Load D parameters into trial_vector[]
    for j in range(D):
        # Perform NP-1 binomial trials.
        if random.uniform(0.0, 1.0) < CR or j == k:
            # Source for trial_vector[j] is a random vector plus weighted differential
            trial_vector[j] = population[individual_index][j] + F * (best_individual[j]-population[individual_index][j]) + F*(population[a][j] - population[b][j])
        else:
            # or trial_vector parameter comes from population[individual_index][j] itself.
            trial_vector[j] = population[individual_index][j]



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    from FixedMess import FixedMes
    NP = FixedMes.populationnumber
    F = FixedMes.F
    CR = FixedMes.CR
    PLS = FixedMes.PLS
    iter = FixedMes.ge
    instance1 = np.load(f'biaozhun_8_11.npy', allow_pickle=True)[0]

    Init = InitM("dis.csv")
    FixedMes.distance = Init.readDis()

    FixedMes.act_info = Init.readData(0, instance1)
    run_diferential_evolution_for_JZJ(FixedMes.act_info,iter,NP,F,CR,PLS)