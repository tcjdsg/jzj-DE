import time
from docplex.cp.model import CpoModel
from docplex.cp.expression import *

import numpy as np
from docplex.cp.modeler import pulse
from docplex.cp.parameters import CpoParameters

from FixedMess import FixedMes
from calFitness import *
from initm import InitM
from utils.visualize.draw import *



class Solution:
    def __init__(self):
        self.nbTasks = 0
        self.numOp = 0
        self.nbProfil = 0
        self.start_time = []
        self.duration_time = []
        self.end_time = []
        self.modes = []
        self.op_profile = []
        self.op_tasks = []
        self.succ = []
        self.fixed_tasks = []
        self.id=[]


import csv


def solution_to_file(sol, filename, seconds):
    output_file_name = f"{filename}_sol.csv"

    with open(output_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["solvetime", "", seconds])
        writer.writerow(["NumOp", "", sol.numOp])
        writer.writerow(["tasks[i]", "Start", "Duration", "End", "Mode"])

        for i in range(sol.nbTasks):
            writer.writerow([
                i,
                sol.start_time[i],
                sol.duration_time[i],
                sol.start_time[i] + sol.duration_time[i],
                sol.modes[i]
            ])

    print("solution saved")

def succ_extension(tasks, sol):
    all_tasks = list(tasks)

    for task in tasks:
        for j in range(len(sol.succ)):
            if task in sol.succ[j] and j not in all_tasks:
                all_tasks.append(j)
            elif j == task:
                for k in range(len(sol.succ[task])):
                    if sol.succ[j][k] not in all_tasks:
                        all_tasks.append(sol.succ[j][k])

    all_tasks.sort()
    return all_tasks

def unchanged_tasks(flexible_tasks, nbTasks):
    return_tasks = []
    j = 0
    for i in range(nbTasks):
        if flexible_tasks[j] == i:
            j += 1
        else:
            return_tasks.append(i)
    return return_tasks

import random

def random_tasks_selector(tasks, percentage):
    tmp1 = list(tasks)
    tmp2 = []
    max_index = int(len(tasks) * (percentage / 100))
    random.shuffle(tmp1)
    for i in range(max_index):
        tmp2.append(tmp1[i])
    return tmp2

def display_vector(v):
    for i in range(len(v)):
        print(v[i], end=" ")
    print()

def display_vector_float(v):
    for i in range(len(v)):
        print(v[i], end=" ")
    print()


def rcpsp_solve(jzj_number,order_number,maxtime,act_info,computing_time):
    sol = Solution()
    best_sol = Solution()
    sol.numOp = 0
    global_time_limit = computing_time

    current_time = time.time()
    failLimit = 30000000
    model = CpoModel(name='JZJ')

    nbTasks = jzj_number*order_number+2#任务数量
    nb_gongwei = 20#保障工位数量
    nb_profil = 4#保障人员类型
    n_stations = [len(FixedMes.constraintS_JZJ[i]) for i in range(5)]#5类保障设备
    TT = maxtime#最大完工时间

    areas = [[] for _ in range(nb_gongwei)]#定义工位

    stations = [[[] for _ in range(n_stations[i])]for i in range(len(n_stations))]#定义设备
    profils = [pulse(interval_var(), 0) for _ in range(nb_profil)]#定义保障人员
    numOp = [model.integer_var() for _ in range(nb_profil)]

    tasks = [model.interval_var(name="task_{}".format(i)) for i in range(nbTasks)]

    ends = []
    sol.succ = [[] for _ in range(nbTasks)]
    choices = []

    for i in range(nbTasks):
        choice = []
        task = tasks[i]
        task.set_length_min(int(act_info[i].duration))#设置保障时长
        task.set_length_max(int(act_info[i].duration))

        sol.succ[i] = act_info[i].successor#
        nbSucc = len(sol.succ[i])
        #设置工序的紧前紧后约束
        for j in range(nbSucc):
            indexj = sol.succ[i][j]
            if indexj < nbTasks:
                model.add(model.end_before_start(task, tasks[indexj]))

        #飞机停机位
        pos_id = act_info[i].belong_plane_id
        if act_info[i].Space > 0:
            areas[pos_id - 1].append(task)

        if act_info[i].RequestStationType >= 0 and act_info[i].duration>0:
            stype = int(act_info[i].RequestStationType)
            for s in range(n_stations[stype]):
                if pos_id in FixedMes.constraintS_JZJ[stype][s]:
                    name = "station_" + str(stype) + "_" + str(s) + "_" + str(i)
                    alt = model.interval_var(optional=True, name=name, size=int(act_info[i].duration))
                    choice.append(alt)

                    stations[stype][s].append(alt)
            model.add(model.alternative(task, choice))

        type = act_info[i].resourceHType
        for j in range(nb_profil):
            if type == j and act_info[i].duration>0:
                profils[j] += pulse(task, int(act_info[i].needH))

        ends.append(model.end_of(task))
        choices.append(choice)

    # 添加约束条件
    for ops in areas:
        if len(ops)>0:
            model.add(model.no_overlap(ops))

    for i in range(len(stations)):
        for j in range(len(stations[i])):
            if len(stations[i][j])>0:
                model.add(model.no_overlap(stations[i][j]))


    n = FixedMes.total_Huamn_resource
    for j in range(nb_profil):
        model.add(profils[j] <= n[j])

    model.add(model.max(ends) <= TT)

  #  model.add(model.minimize(model.sum(numOp)))
    model.add(model.minimize(model.max(ends)))

    # 创建参数对象
    params = CpoParameters()
    # 设置参数值
    params.set_LogVerbosity('Quiet')
    # params.set_log_period(100000)  # 如果需要设置LogPeriod参数，请取消注释此行并替换为实际值
    params.set_FailLimit(failLimit)  # 替换为实际的failLimit值
    params.set_TimeLimit(global_time_limit)  # 替换为实际的global_time_limit值
    params.set_TimeMode('ElapsedTime')

    # 将参数对象与模型关联
    model.set_parameters(params)
    # 创建CpoSolver对象并求解模型
    solver = model.solve()
    end_time = time.time()
    cal  = end_time-current_time
    if 1:
            print(cal)
            #
            # # 输出结果
            # print("Solution: ")
            # solver.print_solution()  # 解的打印
            print("Objective_value")
            print(solver.get_objective_values())
            print(solver.get_solve_status())
            # for j in range(nb_profil):
            #     print("Profil \t:", solver.get_value(numOp[j]))
            # for i in range(nbTasks):
            print(solver.get_value(tasks[2]))

    sol.nbTasks = nbTasks
    sol.nbProfil = nb_profil
    sol.op_profile = [solver[numOp[j]] for j in range(nb_profil)]

    sol.start_time = []
    sol.end_time = []
    for i in range(nbTasks):
        sol.start_time.append(solver.get_value(tasks[i])[0])
        sol.end_time.append(solver.get_value(tasks[i])[1])
        act_info[i].es = sol.start_time[-1]
        act_info[i].ef = sol.end_time[-1]
        # if act_info[i].belong_plane_id==15:
        #         print()

        if len(choices[i]) > 0:
            # print(len(modes[i]))

            for a in range(len(choices[i])):
                # print(solver.get_last_result()[modes[i][a]])
                if len(solver.get_value(choices[i][a])) > 0:
                    # print(solver.get_value(choices[i][a]))
                    # print(choices[i][a].get_name())
                    s_id = choices[i][a].get_name()[11:12]
                    if s_id == "_":
                        s_id = choices[i][a].get_name()[10:11]
                    else:
                        s_id = choices[i][a].get_name()[10:12]
                    sol.modes.append(a)
                    sol.id.append(int(s_id))
                    act_info[i].SheiBei.append([int(choices[i][a].get_name()[8:9]),sol.id[-1]])
        else:
            sol.modes.append(-1)
            sol.id.append(-1)

    return solver.get_objective_values(),cal,solver.get_solve_status()

def allcolate(activities):

        acts = []
        for (key,order) in activities.items():
            acts.append(order)
        acts = sorted(acts,key=lambda x:x.es)
        humans, stations, spaces = initMess()
        humans_resources = [len(humans[i]) for i in range(len(humans))]
        used_humans = [[0] * 200 for i in range(4)]
      #  MTRCA(stations, individual.codes, self.activities)
        for order in acts:
            h_type = order.resourceHType
            need = order.needH
            now_pos = order.belong_plane_id
            t = order.es
            al_h_num,recoed_h = judgeHuman(humans, h_type, need, now_pos, t)
            assert al_h_num >= need
            allocationHuman(recoed_h, humans, activities, order.id, now_pos)
            if len(activities[order.id].SheiBei)>0:
                s_type = activities[order.id].SheiBei[0][0]
                nu = activities[order.id].SheiBei[0][1]
              #  print(s_type,nu)
                stations[s_type][nu].update(order)
        draw_h_s(humans,stations,8)

instances = np.load(f'dataset/DE_8_12_instance4.npy', allow_pickle=True)
Init = InitM("dis.csv")
FixedMes.distance = Init.readDis()

cals = []
sols = []
values = []
if_optimal = []
for i in range(1):
    print("--------",i,"------------")
    FixedMes.act_info = Init.readData(0, instances[i])

    FixedMes.total_Huamn_resource = instances[i][2]
    sol, cal, status = rcpsp_solve(8,12,60,FixedMes.act_info, 60)
    allcolate(FixedMes.act_info)
   # print(status)
    sols.append([sol[0], status])
    values.append(sol[0])
    cals.append(cal)

# np.save(f'time_8586_8_12.npy', cals)
# print(sum(values)/len(values))
# print(sum(cals)/len(cals))




