import copy
import time
from typing import Union, List

import numpy as np
from pyDOE import lhs

from scipy.stats import cauchy
from copy import deepcopy
from Chromo import Chromosome
from calFitness import Fitness
from initm import InitM
from utils import *
from utils.history import History
from utils.logger import Logger


class JADE():
    """
    The original version of: Differential Evolution (JADE)

    Links:
        1. https://doi.org/10.1109/TEVC.2009.2014613

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + miu_f (float): [0.4, 0.6], initial adaptive f, default = 0.5
        + miu_cr (float): [0.4, 0.6], initial adaptive cr, default = 0.5
        + pt (float): [0.05, 0.2], The percent of top best agents (p in the paper), default = 0.1
        + ap (float): [0.05, 0.2], The Adaptation Parameter control value of f and cr (c in the paper), default=0.1

    Examples
    ~~~~~~~~
    # >>> import numpy as np
    # >>> import  DE
    # >>>
    # >>> def objective_function(solution):
    # >>>     return np.sum(solution**2)
    # >>>
    # >>> problem_dict = {
    # >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    # >>>     "minmax": "min",
    # >>>     "obj_func": objective_function
    # >>> }
    # >>>
    # >>> model = DE.JADE(epoch=1000, pop_size=50, miu_f = 0.5, miu_cr = 0.5, pt = 0.1, ap = 0.1)
    # >>> g_best = model.solve(problem_dict)
    # >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    # >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhang, J. and Sanderson, A.C., 2009. JADE: adaptive differential evolution with optional
    external archive. IEEE Transactions on evolutionary computation, 13(5), pp.945-958.
    """

    def __init__(self, activities,epoch: int = 10000, pop_size: int = 100, miu_f: float = 0.5,
                 miu_cr: float = 0.5, pt: float = 0.1, ap: float = 0.1, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_f (float): initial adaptive f, default = 0.5
            miu_cr (float): initial adaptive cr, default = 0.5
            pt (float): The percent of top best agents (p in the paper), default = 0.1
            ap (float): The Adaptation Parameter control value of f and cr (c in the paper), default=0.1
        """
        super().__init__(**kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.miu_f = miu_f
        self.miu_cr = miu_cr
        # np.random.uniform(0.05, 0.2) # the x_best is select from the top 100p % solutions
        self.pt = pt
        # np.random.uniform(1/20, 1/5) # the adaptation parameter control value of f and cr
        self.ap = ap

        self.np1 = self.pop_size/2
        self.np2 = self.pop_size / 2
        self.PSP = 0.5

        self.sort_flag = False
        self.generator = np.random.default_rng(12345)
        self.activities = activities
        self.D = len(list(activities.keys()))

        self.population = [Chromosome([0.0 for _ in range(self.D)], 0.0) for _ in range(self.pop_size)]
        self.individuals_fitness = [0.0] * self.pop_size
        self.lower_bound = [-1] * self.D
        self.upper_bound = [1] * self.D
        self.history = History(log_to=None, log_file='log_file.txt')
        self.logger = Logger(None, log_file='log_file.txt').create_logger(name=f"{self.__module__}.{self.__class__.__name__}")

    def initialize_variables(self):
        self.dyn_miu_cr = self.miu_cr
        self.dyn_miu_f = self.miu_f
        self.dyn_pop_archive = list()


    ### Survivor Selection
    def lehmer_mean(self, list_objects):
        temp = np.sum(list_objects)
        return 0 if temp == 0 else np.sum(list_objects ** 2) / temp

    def RUN(self, total_Huamn_resource):
        self.initialize_variables()

        print("各类型人员组成: ", total_Huamn_resource)
        self.initialize_individuals_randomly()

        Emax = []
        epoch=0
        while epoch < self.epoch:
            epoch += 1
            time_epoch = time.perf_counter()
            self.evolve(epoch)
            time_epoch = time.perf_counter() - time_epoch
            self.track_optimize_step(self.population, epoch, time_epoch)
            self.population = sorted(self.population,key=lambda x:x.fitness)
            print("--{}----------best: {}-----".format(epoch,  self.population[0].fitness))
            self.track_optimize_step(self.population,epoch,time_epoch)



    def track_optimize_step(self, pop: List[Chromosome] = None, epoch: int = None, runtime: float = None) -> None:
        """
        Save some historical data and print out the detailed information of training process in each epoch

        Args:
            pop: the current population
            epoch: current iteration
            runtime: the runtime for current iteration
        """
        ## Save history data

        self.history.list_epoch_time.append(runtime)
        self.history.list_global_best_fit.append(self.history.list_global_best[-1].fitness)
        self.history.list_current_best_fit.append(self.history.list_current_best[-1].fitness)
        # Save the exploration and exploitation data for later usage
        pos_matrix = np.array([agent.codes for agent in pop])
        div = np.mean(np.abs(np.median(pos_matrix, axis=0) - pos_matrix), axis=0)
        self.history.list_diversity.append(np.mean(div, axis=0))
        ## Print epoch
        self.logger.info(f">>>Epoch: {epoch}, Current best: {self.history.list_current_best[-1].fitness}, "
                         f"Global best: {self.history.list_global_best[-1].fitness}, Runtime: {runtime:.5f} seconds")

    def initialize_individuals_randomly(self):
        # uniform(lower_bound[j], upper_bound[j])

        for j in range(self.D):
            # 拉丁超立方采样
            xx = self.lower_bound[j] + ( self.upper_bound[j] -  self.lower_bound[j]) * lhs(1,  self.pop_size)
            for i in range(self.pop_size):
                self.population[i].codes[j] = xx[i]
        for i in range(self.pop_size):
            self.individuals_fitness[i] = Fitness(self.population[i].codes, self.activities, 'l')
            self.population[i].fitness = self.individuals_fitness[i]
    # TODO
    # def local_search(self):


    def mutation1(self,  idx, f, cr):
        new_pop = self.population + self.dyn_pop_archive
        idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
        while True:
            x_r2 = new_pop[self.generator.integers(0, len(new_pop))]
            if np.any(x_r2.codes - self.population[idx_list[0]].codes) and np.any(x_r2.codes - self.population[idx].codes) and np.any(x_r2.codes - self.population[idx_list[1]].codes):
                break

        x_new = self.population[idx].codes + f*(self.population[idx_list[0]].codes - self.population[idx].codes+
                    self.population[idx_list[1]].codes - x_r2.codes)

        pos_new = np.where(self.generator.random(self.D) < cr, x_new, self.population[idx].codes)
        j_rand = self.generator.integers(0, self.D)
        pos_new[j_rand] = x_new[j_rand]
        #  pos_new = self.correct_solution(pos_new)
        return pos_new

    def mutation2(self,  idx, f, cr):


        idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
        x_new = self.population[idx].codes + f*(self.population[idx_list[0]].codes - self.population[idx].codes+
                    self.population[idx_list[1]].codes - self.population[idx_list[2]].codes)

        pos_new = np.where(self.generator.random(self.D) < cr, x_new, self.population[idx].codes)
        j_rand = self.generator.integers(0, self.D)
        pos_new[j_rand] = x_new[j_rand]
        #  pos_new = self.correct_solution(pos_new)
        return pos_new



    def mutation3(self,pop_sorted,idx,f,cr):

            top = int(self.pop_size * self.pt)
            x_best = pop_sorted[self.generator.integers(0, top)]
            x_r1 = self.population[self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))]
            new_pop = self.population + self.dyn_pop_archive
            n_dims = len(self.population[idx].codes)
            while True:
                x_r2 = new_pop[self.generator.integers(0, len(new_pop))]
                if np.any(x_r2.codes - x_r1.codes) and np.any(x_r2.codes - self.population[idx].codes):
                    break

            x_new = self.population[idx].codes + f * (x_best.codes - self.population[idx].codes +
                        x_r1.codes - x_r2.codes)

            pos_new = np.where(self.generator.random(n_dims) < cr, x_new, self.population[idx].codes)
            j_rand = self.generator.integers(0, n_dims)
            pos_new[j_rand] = x_new[j_rand]
            return pos_new
    def mutation4(self,pop_sorted,idx,f,cr):
        top = int(self.pop_size * self.pt)
        x_best = pop_sorted[self.generator.integers(0, top)]
        idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
        x_new = x_best.codes + f * (
                self.population[idx_list[0]].codes - self.population[idx_list[1]])

        pos_new = np.where(self.generator.random(self.D) < cr, x_new, self.population[idx].codes)
        j_rand = self.generator.integers(0, self.D)
        pos_new[j_rand] = x_new[j_rand]
        return pos_new

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        list_f = list()
        list_cr = list()
        temp_f = list()
        temp_cr = list()
        pop_sorted = get_sorted_population(self.population)
        pop = []
        if epoch == self.PSP:
            self.np1 = self.pop_size/2
            self.np2 = self.pop_size/2
        for idx in range(0, self.pop_size):
            ## Calculate adaptive parameter cr and f
            cr = self.generator.normal(self.dyn_miu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            while True:
                f = cauchy.rvs(self.dyn_miu_f, 0.1)
                if f < 0:
                    continue
                elif f > 1:
                    f = 1
                break
            temp_f.append(f)
            temp_cr.append(cr)
            if epoch < self.PSP:
                if idx < self.np1 :
                    pos_new = self.mutation1(idx, f, cr)
                if idx >= self.np1 :
                    pos_new = self.mutation2(idx, f, cr)

            if epoch >= self.PSP:

                if idx < self.np1 :
                    pos_new = self.mutation3(pop_sorted,idx, f, cr)
                if idx >= self.np1 :
                    pos_new = self.mutation4(pop_sorted,idx, f, cr)

          #  pos_new = self.correct_solution(pos_new)
            agent = Chromosome(pos_new, 0)
            pop.append(agent)

            pop[-1].fitness = Fitness(agent.codes, copy.deepcopy(self.activities),'l')

        #pop = self.update_target_for_population(pop)
        f_best_old_1 = sorted(self.population[:self.np1],key=lambda x:x.fitness)[0].fitness
        f_best_old_2 = sorted(self.population[self.np1:],key=lambda x:x.fitness)[0].fitness

        f_best_new_1 = sorted(pop[:self.np1],key=lambda x:x.fitness)[0].fitness
        x_best_1 = sorted(pop[:self.np1],key=lambda x:x.fitness)[0]

        f_best_new_2 = sorted(pop[self.np1:],key=lambda x:x.fitness)[0].fitness
        x_best_2 = sorted(pop[self.np1:], key=lambda x: x.fitness)[0]

        IQ1 = (f_best_old_1 - f_best_new_1)/f_best_old_1
        IQ2 = (f_best_old_2 - f_best_new_2)/f_best_old_2
        DIV1 = 0
        DIV2 = 0

        for idx in range(0, self.pop_size):
            if idx < self.np1:
                DIV1 += np.linalg.norm(pop[idx].codes - x_best_1)
            else:
                DIV2 += np.linalg.norm(pop[idx].codes - x_best_2)
            if pop[idx].fitness < self.population[idx].fitness:
                #把差的解存档
                self.dyn_pop_archive.append(copy.deepcopy(self.population[idx]))
                list_cr.append(temp_cr[idx])
                list_f.append(temp_f[idx])
                self.population[idx] = copy.deepcopy(pop[idx])
        NDIV1 = DIV1/ (DIV1 + DIV2)
        NDIV2 = DIV2 / (DIV1 + DIV2)
        NV1 = ((1-IQ1)+NDIV1)/((1-IQ1)+(1-IQ2)+NDIV1+NDIV2)
        NV2 = ((1-IQ2)+NDIV2)/((1-IQ1)+(1-IQ2)+NDIV2+NDIV1)
        self.np1 = max(0.1,min(0.9,NV1))*self.pop_size
        self.np2 = self.pop_size-self.np1

        # Randomly remove solution
        temp = len(self.dyn_pop_archive) - self.pop_size
        if temp > 0:
            idx_list = self.generator.choice(range(0, len(self.dyn_pop_archive)), temp, replace=False)
            archive_pop_new = []
            for idx, solution in enumerate(self.dyn_pop_archive):
                if idx not in idx_list:
                    archive_pop_new.append(solution)
            self.dyn_pop_archive = deepcopy(archive_pop_new)
        # Update miu_cr and miu_f
        if len(list_cr) == 0:
            self.dyn_miu_cr = (1 - self.ap) * self.dyn_miu_cr + self.ap * 0.5
        else:
            self.dyn_miu_cr = (1 - self.ap) * self.dyn_miu_cr + self.ap * np.mean(np.array(list_cr))
        if len(list_f) == 0:
            self.dyn_miu_f = (1 - self.ap) * self.dyn_miu_f + self.ap * 0.5
        else:
            self.dyn_miu_f = (1 - self.ap) * self.dyn_miu_f + self.ap * self.lehmer_mean(np.array(list_f))

if __name__ == '__main__':
    from FixedMess import FixedMes
    NP = FixedMes.populationnumber
    F = FixedMes.F
    CR = FixedMes.CR
    PLS = FixedMes.PLS
    iter = FixedMes.ge
    instance1 = np.load(f'biaozhun_12_11.npy', allow_pickle=True)[0]

    Init = InitM("dis.csv")
    FixedMes.distance = Init.readDis()

    FixedMes.act_info = Init.readData(0, instance1)
    model = JADE(activities=FixedMes.act_info, epoch=1000, pop_size=50, miu_f = 0.5, miu_cr = 0.5, pt = 0.1, ap = 0.1)

