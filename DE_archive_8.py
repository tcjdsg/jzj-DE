import copy
import random
import time
from typing import Union, List
import numpy
from pyDOE import lhs

from scipy.stats import cauchy
from copy import deepcopy
from Chromo import Chromosome
from calFitness import *
from initm import InitM
from utilss import *
from utils.history import History
from utils.logger import Logger
from utils.visualize.draw import draw_h_s


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

    def __init__(self, activities,zaibian:int=2,cmax:int=90,psp:float = 1.0,epoch: int = 10000, pop_size: int = 100,
                 M: int = 100,
                 E: float = 0.01,
                 miu_f: float = 0.5,
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
        self.CMAX = cmax
        self.epoch = epoch
        self.pop_size = pop_size
        self.miu_f = miu_f
        self.miu_cr = miu_cr
        # np.random.uniform(0.05, 0.2) # the x_best is select from the top 100p % solutions
        self.pt = pt
        # np.random.uniform(1/20, 1/5) # the adaptation parameter control value of f and cr
        self.ap = ap
        self.zaibian=zaibian
        self.M = M


        self.np1 = int(self.pop_size/2)
        self.np2 = int(self.pop_size / 2)
        self.PSP = int(psp * self.epoch)

        self.badGen=0
        self.a = E
        self.sort_flag = False
        self.generator = numpy.random.default_rng(2345)
        self.activities = activities
        self.D = len(list(activities.keys()))

        self.population = [Chromosome([0.0 for _ in range(self.D)], 0.0) for _ in range(self.pop_size)]
        self.individuals_fitness = [0.0] * self.pop_size
        self.lower_bound = [-1] * self.D
        self.upper_bound = [1] * self.D
        self.history = History(log_to=None, log_file='log_file.txt')
        self.logger = Logger(None, log_file='log_file.txt').create_logger(name=f"{self.__module__}.{self.__class__.__name__}")

        self.ES = [0 for _ in range(len(list(self.activities.keys())))]
        self.LF = [self.CMAX for _ in range(len(list(self.activities.keys())))]
        dfsEST(self.activities,len(list(self.activities.keys()))-1,self.ES)
        dfsLFT(self.activities,0,self.LF,self.CMAX)

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
        print("--{}----------best: {}-----".format(0, self.population[0].fitness))

        epoch=0
        start=time.time()
        while epoch < self.epoch:
            epoch += 1
            time_epoch = time.perf_counter()

            self.evolve(epoch)
            time_epoch = time.perf_counter() - time_epoch
            self.track_optimize_step(self.population, epoch, time_epoch)
            self.population = sorted(self.population, key=lambda x:x.fitness)

            print("--{}----------best: {}-----".format(epoch,  self.population[0].fitness))
            self.track_optimize_step(self.population,epoch,time_epoch)
        end = time.time()
        print(end-start)
        # self.local_search(self.population[0])

        # self.allcolate(self.population[0], 'r')
        # list_order = encoder(self.population[0].codes, self.activities)
        # acts, codes = dfs_serialGenerationScheme(self.activities, list_order)
        # print("---------------",acts[len(list_order)-1].ef)

    def Repetition(self):
        orders = np.load('orders.npy')
        o_humans = np.load('orders_human.npy',allow_pickle=True).item()
        o_stations = np.load('orders_station.npy', allow_pickle=True).item()
        humans, stations, spaces = initMess()

        for i in range(orders.shape[0]):
            id = orders[i][0]
            order = self.activities[id]
            h_type = order.resourceHType
            need = order.needH
            now_pos = order.belong_plane_id

            order.es = orders[i][1]
            order.ef = orders[i][2]
            t = order.es
            if order.ef-order.es > 0:
                al_h_num, recoed_h = judgeHuman(humans, h_type, need, now_pos, t)
                assert al_h_num >= need
                order_humans = o_humans[id]
                k=0
                while need>0:
                    index = order_humans[k][1]
                    humans[h_type][index].update(order)
                    k+=1
                    need-=1

                if o_stations[id][0] > 0:
                    s_type = o_stations[id][0]
                    nu = o_stations[id][1]
                    stations[s_type][nu].update(order)

        draw_h_s(humans, stations, 8)
    def allcolate(self, individual,lr):
        orders = []

        individual.fitness,_ = Fitness(individual.codes,self.activities,lr)
        acts = []
        for (key,order) in self.activities.items():
            acts.append(order)
        acts = sorted(acts,key=lambda x:x.es)
        humans, stations, spaces = initMess()
        humans_resources = [len(humans[i]) for i in range(len(humans))]
        used_humans = [[0] * 200 for i in range(4)]
      #  MTRCA(stations, individual.codes, self.activities)
        orders_humans = {}
        order_stations = {}
        for order in acts:

            orders.append((order.id,order.es,order.ef))
            h_type = order.resourceHType
            need = order.needH
            now_pos = order.belong_plane_id
            t = order.es
            al_h_num,recoed_h = judgeHuman(humans, h_type, need, now_pos, t)
            assert al_h_num >= need
            if order.ef>order.es:
                orders_humans[order.id] = allocationHuman(recoed_h, humans, self.activities, order.id, now_pos)
                if len(self.activities[order.id].SheiBei)>0:
                    s_type= self.activities[order.id].SheiBei[0][0]
                    nu = self.activities[order.id].SheiBei[0][1]
                    stations[s_type][nu].update(order)
                    order_stations[order.id] = (s_type,nu)
                else:
                    order_stations[order.id] = (-1, -1)
        np.save('orders.npy',orders)
        np.save('orders_human.npy',orders_humans)
        np.save('orders_station.npy',order_stations)
        draw_h_s(humans, stations,8)


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
        self.history.list_global_best_fit.append(self.history.list_global_best[-1])
        self.history.list_current_best_fit.append(self.history.list_current_best[-1])
        # Save the exploration and exploitation data for later usage
        pos_matrix = np.array([agent.codes for agent in pop])
        div = np.mean(np.abs(np.median(pos_matrix, axis=0) - pos_matrix), axis=0)
        self.history.list_diversity.append(np.mean(div, axis=0))
        ## Print epoch
        self.logger.info(f">>>Epoch: {epoch}, Current best: {self.history.list_current_best[-1]}, "
                         f"Global best: {self.history.list_global_best[-1]}, Runtime: {runtime:.5f} seconds")

    def initialize_individuals_randomly(self):
        # uniform(lower_bound[j], upper_bound[j])

        for j in range(self.D):
            # 拉丁超立方采样
            xx = self.lower_bound[j] + (self.upper_bound[j] - self.lower_bound[j]) * lhs(1,  self.pop_size)
            for i in range(self.pop_size):
                self.population[i].codes[j] = xx[i][0]
        for i in range(self.pop_size):
            self.population[i].fitness,self.population[i].codes = Fitness(self.population[i].codes, copy.deepcopy(self.activities), 'l')
           # self.local_search( self.population[i])
    # TODO 并行调度生成机制，然后就可以说明在做计划时，串行调度机制更好。


    # TODO 基于模拟退火机制的插入邻域搜索
    def local_search(self,popone):

        allTasks = copy.deepcopy(self.activities)

        Fitness(popone.codes, allTasks, 'l')
        print("进行右对齐调度之前：{}".format(allTasks[len(list(allTasks.keys())) - 1].ef))
        Fitness(popone.codes, allTasks, 'r')
        print("进行右对齐调度之后：{}".format(allTasks[len(list(allTasks.keys())) - 1].ef))

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
                self.population[idx_list[0]].codes - self.population[idx_list[1]].codes)

        pos_new = np.where(self.generator.random(self.D) < cr, x_new, self.population[idx].codes)
        j_rand = self.generator.integers(0, self.D)
        pos_new[j_rand] = x_new[j_rand]
        return pos_new

    def zaibian_2(self):
        # 种群多样性指标小于
        if len(model.history.list_average)>50:
            av = model.history.list_average[-1]
            best = model.history.list_current_best[-1]
            if abs(av - best)/best <= self.a:
                self.badGen += 1
            if self.badGen >= self.M:
                print("-----发生灾变--------average：{}---best：{}".format(av, best))
                best_pops = copy.deepcopy(sorted(self.population, key=lambda x: x.fitness)[:int(0.1 * self.pop_size)])
                self.initialize_individuals_randomly()
                self.population = best_pops + self.population[:self.pop_size - len(best_pops)]
                self.badGen = 0



    def zaibian_1(self):
        # 当种群中重复个体达到以上时，就进行灾变
        # 重复是指编码空间上的重复，即：
        P = []
        pop_dupli = [0 for _ in range(len(self.population))]
        P.append(self.population[0])
        for i in range(self.pop_size):
            pop = self.population[i]
            for q in P:

                flag = True
                if pop.fitness==q.fitness:
                    for j in range(len(list(self.activities.keys()))):
                        if pop.codes[j]!=q.codes[j]:
                            flag=False
                            break
                else:
                    flag=False
                if flag:
                    pop_dupli[i]+=1
            P.append(pop)
        for i in range(len(pop_dupli)):
            if pop_dupli[i]>=0.5*self.pop_size:#说明有一半以上的重复个体
                best_pops = sorted(self.population,key=lambda x:x.fitness)[0:int(0.1*self.pop_size)]
                self.initialize_individuals_randomly()
                self.population = best_pops + self.population[:self.pop_size - len(best_pops)]
                break

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
        if self.zaibian==1:
            self.zaibian_1()
        elif self.zaibian==2:
            self.zaibian_2()

        random.shuffle(self.population)
        pop_sorted = sorted(self.population,key=lambda x:x.fitness)
        sumf = 0
        for i in range(len(pop_sorted)):
            sumf+=pop_sorted[i].fitness

        self.history.list_current_best.append(pop_sorted[0].fitness)
        self.history.list_average.append(sumf/len(pop_sorted))
        self.history.list_global_best.append(pop_sorted[0].fitness if pop_sorted[0].fitness<self.history.list_current_best[-1] else self.history.list_current_best[-1])
        pop = []
        if epoch == self.PSP:
            self.np1 = int(self.pop_size/2)
            self.np2 = int(self.pop_size/2)
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
            if epoch < self.PSP :
                if idx < self.np1:
                    pos_new = self.mutation1(idx, f, cr)
                if idx >= self.np1:
                    pos_new = self.mutation2(idx, f, cr)

            if epoch >= self.PSP:

                if idx < self.np1 :
                    pos_new = self.mutation3(pop_sorted,idx, f, cr)
                if idx >= self.np1 :
                    pos_new = self.mutation4(pop_sorted,idx, f, cr)


            agent = Chromosome(pos_new, 0)
            pop.append(agent)

            pop[-1].fitness,pop[-1].codes = Fitness(agent.codes, copy.deepcopy(self.activities),'l')

        #pop = self.update_target_for_population(pop)
        f_best_old_1 = sorted(self.population[:self.np1],key=lambda x:x.fitness)[0].fitness
        f_best_old_2 = sorted(self.population[self.np1:],key=lambda x:x.fitness)[0].fitness

        f_best_new_1 = sorted(pop[:self.np1],key=lambda x:x.fitness)[0].fitness
        x_best_1 = sorted(pop[:self.np1],key=lambda x:x.fitness)[0].codes

        f_best_new_2 = sorted(pop[self.np1:],key=lambda x:x.fitness)[0].fitness
        x_best_2 = sorted(pop[self.np1:], key=lambda x: x.fitness)[0].codes

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
        self.np1 = int(max(0.1,min(0.9,NV1))*self.pop_size)
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
        # print("----F----",self.dyn_miu_f)
        # print("----Cr----",self.dyn_miu_cr)

if __name__ == '__main__':
    from FixedMess import FixedMes
    #NP = FixedMes.populationnumber
    # F = FixedMes.F
    # CR = FixedMes.CR
    # PLS = FixedMes.PLS
    # iter = FixedMes.ge
    file = 'dataset/DE_8_12_instance4.npy'
    epoch = 300
    NP = [50,100,150,200]
    PSP = [0,0.25,0.5,0.75]
    M = [200,40,60,80]
    E = [0.02,0.03,0.04,0.05]
    cmax = 70
    zaibian = 2
    instances = np.load(file, allow_pickle=True)

    zuhe = [[1,1,1,1],  #1
            [1,2,2,2],  #2
            [1,3,3,3],  #3
            [1,4,4,4],  #4
            [2,1,2,2],  #5
            [2,2,1,4],  #6
            [2,3,4,1],  #7
            [2,4,3,2],  #8
            [3,2,4,3],  #9
            [3,1,3,4],  #10
            [3,3,1,3],  #11
            [3,4,2,1],  #12
            [4,1,4,2],  #13
            [4,2,3,1],  #14
            [4,3,2,4],  #15
            [4,4,1,3]   #16
            ]

    Init = InitM("dis.csv")
    FixedMes.distance = Init.readDis()
    best_res = []
    mean_res = []

    for i in range(1):
        best_psp = 0
        best_fit = 99999
        instance = instances[i]
        data = None
        FixedMes.act_info = Init.readData(0, instance)
        best_zuhe = []

        FixedMes.total_Huamn_resource = instance[2]
        for j in range(len(zuhe)):
            n = NP[zuhe[j][0]-1]
            psp = PSP[zuhe[j][1]-1]
            m = M[zuhe[j][2]-1]
            e = E[zuhe[j][3] - 1]
            res = 99999
            ress = []
            data_psp = None
            for i in range(10):
                model = JADE(activities=FixedMes.act_info, zaibian=zaibian, cmax=cmax, psp=psp,epoch=epoch, pop_size=n,M = m,E=e, miu_f = 0.6, miu_cr = 0.6, pt = 0.1, ap = 0.1)
                start = time.time()
                model.RUN(FixedMes.total_Huamn_resource)

                end = time.time()

                fit = sorted(model.population, key=lambda x: x.fitness)[0].fitness
                if fit == 56:
                    model.allcolate(model.population[0], 'l')
                    break
                ress.append(fit)
                if fit < res:
                    res = fit
            best_res.append(res)
            mean_res.append(sum(ress)/len(ress))
            if res < best_fit:
                best_fit = res
                best_zuhe = [np, psp, m, e]
        print("bestfit：", best_fit)
        print("best zuhe：", best_zuhe)

    np.save('result\zhengjiao_best_{}.npy'.format(file[10:-4]), best_res)
    np.save('result\zhengjiao_mean_{}.npy'.format(file[10:-4]), mean_res)
        #             data_psp = pd.DataFrame(data= model.history.list_current_best)
        # # PATH为导出文件的路径和文件名
        #             data_psp.to_csv("output\psp\epoch-{}-psp-{}-zaibian-{}-file-{}-popsize-{}.csv".format(epoch,psp,zaibian,file[10:-4],popsize))

        # data2 = pd.DataFrame(data= model.history.list_current_best)
        # # PATH为导出文件的路径和文件名
        # data2.to_csv("output\epoch-{}-file-{}-popsize-{}-average.csv".format(epoch,file[10:-4],popsize))
                # filename="output\epoch-{}-file-{}-popsize-{}-best".format(epoch,file[10:-4],popsize)

                # export_convergence_chart(data=model.history.list_current_best, title=None,
                #                  legend="最优适应度", x_label="迭代次数")
                #
                # export_convergence_chart(data=model.history.list_average, title=None,
                #                   legend="平均适应度", x_label="迭代次数", )
            # if res<best_fit:
            #     best_fit=res
            #     best_psp = psp
            #     data = data_psp
            #     data.to_csv(
            #     "output\psp\epoch-{}-psp-{}-file-{}-popsize-{}-best.csv".format(epoch, best_psp, file[10:-4], popsize))
#现在还没加局部搜索呢

