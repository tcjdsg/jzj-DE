import copy
import time
from docplex.cp.model import CpoModel
from docplex.cp.expression import *

import numpy as np
from docplex.cp.modeler import pulse
from docplex.cp.parameters import CpoParameters

from FixedMess import FixedMes
from initm import InitM

class Solution:
    def __init__(self):
        self.s_id = []
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
    all_tasks = copy.deepcopy(tasks)
  #  print(tasks)

    for i in range(len(tasks)):
        task = tasks[i]
        for j in range(len(sol.succ)):
            if task in sol.succ[j] and j not in all_tasks:
             #   if sol.succ[j].index(task)!=len(sol.succ[j])-1 and all_tasks.index(j)==len(all_tasks)-1:
                    all_tasks.append(j)
            elif j == task:
                for k in range(len(sol.succ[task])):
                    if sol.succ[j][k] not in all_tasks:
                            all_tasks.append(sol.succ[j][k])

    all_tasks.sort()

   # print(all_tasks)
    return all_tasks

def unchanged_tasks(flexible_tasks, nbTasks):
    return_tasks = []
    # [58, 60, 61, 62, 63, 64, 69, 70, 71, 72, 73, 74, 75, 77]
    flexible_tasks.append(-1)
  #  print(flexible_tasks)
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

def find(k,act_info,start_time,end_time,op):
    #k为类别数
    #start_time存放的是每个工序的开始时间，一共有N个工序
    #end_time存放的是每个工序的结束时间，一共有N个工序
    peak_tasks = []
    curr_tasks = []

    tmp1 = [0, 0]
    clean_value = [0 for _ in range(k)]
    n = len(start_time)
    sorted_start = [[] for _ in range(n)]

    for i in range(n):
        tmp1[0] = start_time[i]
        tmp1[1] = i
        sorted_start[i] = copy.deepcopy(tmp1)

    sorted_start = sorted(sorted_start, key=lambda x: x[0])
    for i in range(n):
        flag = False
        tmp = []
        curr_time = sorted_start[i][0]
        c = len(curr_tasks)
        for j in range(c):
            task = curr_tasks[j]
            if end_time[task] > curr_time:
                tmp.append(task)
        curr_tasks = tmp
        curr_tasks.append(sorted_start[i][1])
        curr_value = clean_value
        cc = len(curr_tasks) # cc=c+1
        for j in range(cc):
            k = act_info[curr_tasks[j]].resourceHType
            curr_value[k] += int(act_info[curr_tasks[j]].needH)

        for j in range(k):
            if curr_value[j] == op[j]:
                flag = True
                break
        if flag:
            for j in range(len(curr_tasks)):
                if curr_tasks[j] not in peak_tasks:
                    peak_tasks.append(curr_tasks[j])
    peak_tasks = sorted(peak_tasks)
    return peak_tasks

def find_peak_tasks(sol, act_info):
    peak_tasks = []
    curr_tasks = []

    tmp1 = [0, 0]
    clean_value = [0 for _ in range(sol.nbProfil)]
    sorted_start = [[] for _ in range(len(sol.start_time))]

    for i in range(len(sorted_start)):
        tmp1[0] = sol.start_time[i]
        tmp1[1] = i
        sorted_start[i] = copy.deepcopy(tmp1)
    sorted_start = sorted(sorted_start, key=lambda x: x[0])
    for i in range(len(sorted_start)):
        flag = False
        tmp = []
        curr_time = sorted_start[i][0]
        for j in range(len(curr_tasks)):
            task = curr_tasks[j]
            if sol.end_time[task] > curr_time:
                tmp.append(task)

        curr_tasks = tmp
        curr_tasks.append(sorted_start[i][1])
        curr_value = clean_value
        for j in range(len(curr_tasks)):
            k = act_info[curr_tasks[j]].resourceHType
            curr_value[k] += int(act_info[curr_tasks[j]].needH)

        for j in range(sol.nbProfil):
            if curr_value[j] == sol.op_profile[j]:
                flag = True
                break
        if flag:
            for j in range(len(curr_tasks)):
                if curr_tasks[j] not in peak_tasks:
                    peak_tasks.append(curr_tasks[j])
    peak_tasks = sorted(peak_tasks)
    return peak_tasks

def rcpsp_solve(jzj_number,order_number,maxtime,act_info,computing_time, save_sol, filename):
    sol = Solution()
    best_sol = Solution()
    sol.numOp = 0
    global_time_limit = computing_time
    tt =time.time()

    current_time = time.time()
    failLimit = 30000000
    model = CpoModel(name='JZJ')

    nbTasks = jzj_number * order_number + 2
    nb_gongwei = 16
    nb_profil = 4
    n_stations = [len(FixedMes.constraintS_JZJ[i]) for i in range(5)]#5类保障设备
    TT_1 = maxtime

    areas = [[] for _ in range(nb_gongwei)]  # 定义工位

    stations = [[[] for _ in range(n_stations[i])] for i in range(len(n_stations))]  # 定义设备
    profils = [pulse(interval_var(), 0) for _ in range(nb_profil)]  # 定义保障人员
    numOp = [model.integer_var() for _ in range(nb_profil)]

    tasks = [model.interval_var(name="task_{}".format(i)) for i in range(nbTasks)]
    modes = [interval_var_list(0) for _ in range(nbTasks)]
    ends = []
    sol.succ = [[] for _ in range(nbTasks)]

    presMode = [CpoExpr(interval_var(), "mode") for _ in range(nbTasks)]

    choices = []
    for i in range(nbTasks):
        choice = []
        task = tasks[i]
        task.set_length_min(int(act_info[i].duration))
        task.set_length_max(int(act_info[i].duration))

        sol.succ[i] = act_info[i].successor
        nbSucc = len(sol.succ[i])
        for j in range(nbSucc):
            indexj = sol.succ[i][j]
            if indexj < nbTasks:
                model.add(model.end_before_start(task, tasks[indexj]))
            # 飞机停机位
        pos_id = act_info[i].belong_plane_id
        if act_info[i].Space > 0:
            areas[pos_id - 1].append(task)

        if act_info[i].RequestStationType >= 0:
            pp = 0
            k = 0
            stype = int(act_info[i].RequestStationType)
            for s in range(n_stations[stype]):
                if pos_id in FixedMes.constraintS_JZJ[stype][s]:
                    name = "station_" + str(stype) + "_" + str(s) + "_" + str(i)
                    alt = model.interval_var(optional=True, name=name, size=int(act_info[i].duration))
                    choice.append(alt)
                    stations[stype][s].append(alt)

                    if pp == 0:
                        presMode[i] = (s + 1) * (model.presence_of(alt))
                        pp = 1
                    else:
                        presMode[i] += (s + 1) * (model.presence_of(alt))
            model.add(presMode[i]>=1)
            model.add(model.alternative(task, choice))

        type = act_info[i].resourceHType
        for j in range(nb_profil):
            if type == j:
                profils[j] += pulse(task, int(act_info[i].needH))

        ends.append(model.end_of(task))
        choices.append(choice)

    # 添加约束条件
    for ops in areas:
        if len(ops) > 0:
            model.add(model.no_overlap(ops))

    for i in range(len(stations)):
        for j in range(len(stations[i])):
            if len(stations[i][j]) > 0:
                model.add(model.no_overlap(stations[i][j]))

    n = [4, 4, 8, 9]
    for j in range(nb_profil):
        model.add(profils[j] <= numOp[j])

    model.add(model.max(ends) <= TT_1)
    model.add(model.minimize(model.sum(numOp)))
    #  model.add(model.minimize(model.max(ends)))

    # 创建参数对象
    params = CpoParameters()
    current_time = time.time()
    elapsed_secs = current_time - tt
    # 设置参数值
    params.set_LogVerbosity('Quiet')
    # params.set_log_period(100000)  # 如果需要设置LogPeriod参数，请取消注释此行并替换为实际值
    params.set_FailLimit(failLimit)  # 替换为实际的failLimit值
    params.set_TimeLimit(global_time_limit - elapsed_secs)  # 替换为实际的global_time_limit值
    params.set_TimeMode('ElapsedTime')

    # 将参数对象与模型关联
    model.set_parameters(params)
    # 创建CpoSolver对象并求解模型
    solver = model.start_search()
    solver.next()
    end_time = time.time()
    cal  = end_time-current_time
    for j in range(nb_profil):
                print("Profil \t:", solver.get_last_solution()[numOp[j]])
    # if 1:
    #         print(cal)
    #         # 输出结果
    #         print("Solution: ")
    #         print(solver.get_last_solution())  # 解的打印
    #         # print("Objective_value")
    #         # print(solver.get_objective_values())
    #         for j in range(nb_profil):
    #             print("Profil \t:", solver.get_last_solution()[numOp[j]])
    #         # for i in range(nbTasks):
    #         #     print(solver.get_value(tasks[i]))

    sol.nbTasks = nbTasks
    sol.nbProfil = nb_profil
    sol.op_profile = [solver.get_last_solution()[numOp[j]] for j in range(nb_profil)]

    sol.numOp = sum(sol.op_profile)

    sol.start_time = [solver.get_last_result()[tasks[i]][0] for i in range(nbTasks)]
 #   print(sol.start_time)
    sol.end_time = [solver.get_last_result()[tasks[i]][1] for i in range(nbTasks)]
    sol.duration_time = [solver.get_last_result()[tasks[i]][2] for i in range(nbTasks)]
    for i in range(nbTasks):
       if len(choices[i]) > 0:
          # print(len(modes[i]))
           for a in range(len(choices[i])):
              # print(solver.get_last_result()[modes[i][a]])
               if len(solver.get_last_result()[choices[i][a]]) > 0:
                   s_id = choices[i][a].get_name()[11:12]
                   if s_id=="_":
                       s_id = choices[i][a].get_name()[10:11]
                   else:
                       s_id = choices[i][a].get_name()[10:12]
                   sol.modes.append(a)
                   sol.s_id.append(int(s_id))
       else:
           sol.modes.append(-1)
           sol.s_id.append(-1)

    # for i in range(nbTasks):
    #     print(i)
    #     print(sol.modes[i])

    best_sol = sol
    count=0
    time_increment_count = 0
    fixation_rate = 100

    base_time = 10
    time_increment = 10
    fix_rate_decrement = 10

    current_time = time.time()
    elapsed_secs = current_time - tt

    kk = 0
    while elapsed_secs <= global_time_limit:
        print("-----------------",kk,"-------------------")
        kk += 1
        cst = []
        flexible_tasks = find_peak_tasks(sol, act_info)
        #print(flexible_tasks)
        flexible_tasks = succ_extension(flexible_tasks, sol)
        flexible_tasks = unchanged_tasks(flexible_tasks, nbTasks)

        flexible_tasks = random_tasks_selector(flexible_tasks, fixation_rate)
        flexible_tasks = sorted(flexible_tasks)

        for i in range(len(flexible_tasks)):
            tasks[flexible_tasks[i]].set_start_max(sol.start_time[flexible_tasks[i]])
            tasks[flexible_tasks[i]].set_start_min(sol.start_time[flexible_tasks[i]])
       #     print(flexible_tasks[i])
            if sol.modes[flexible_tasks[i]] >= 1:
                tmp_cst = (presMode[flexible_tasks[i]] == sol.modes[flexible_tasks[i]])
                model.add(tmp_cst)
                cst.append(tmp_cst)

        bound_cst = (model.sum(numOp) < best_sol.numOp)
        model.add(bound_cst)

        current_time = time.time()
        elapsed_secs = current_time-tt
   #     params.set_TimeLimit(min(max(0,int(global_time_limit-elapsed_secs)),base_time+time_increment_count*time_increment))
        model.get_parameters().set_TimeLimit(min(max(0,int(global_time_limit-elapsed_secs)), base_time+time_increment_count*time_increment))
        # print(model.get_all_variables())
        # print(model.get_all_expressions())
        solver = model.solve()
       # solver.next()
        print("solve_Time: ",solver.solveTime)
        print("Solve_status:  ",solver.get_solve_status())
        print("n_profiles",[solver[numOp[j]] for j in range(nb_profil)])

        if solver.get_solve_status() == "Optimal" or solver.get_solve_status()=="Feasible":
            print("successssssssss")
            sol.nbTasks = nbTasks
            sol.nbProfil = nb_profil
            sol.op_profile = [solver[numOp[j]] for j in range(nb_profil)]
            sol.numOp = sum(sol.op_profile)
            sol.start_time = [solver["task_"+str(i)][0] for i in range(nbTasks)]
            #print(sol.start_time)
            sol.end_time = [solver["task_"+str(i)][1]  for i in range(nbTasks)]
            sol.duration_time = [solver["task_"+str(i)][2]  for i in range(nbTasks)]
            sol.modes = []
            sol.s_id = []
            for i in range(nbTasks):
                if len(modes[i]) > 0:
                    # print(len(modes[i]))
                    for a in range(len(modes[i])):
                        # print(solver.get_last_result()[modes[i][a]])
                        if len(solver.get_value(modes[i][a])) > 0:
                            s_id = choices[i][a].get_name()[11:12]
                            if s_id == "_":
                                s_id = choices[i][a].get_name()[10:11]
                            else:
                                s_id = choices[i][a].get_name()[10:12]
                            sol.modes.append(a)
                            sol.s_id.append(int(s_id))
                else:
                    sol.modes.append(-1)
                    sol.s_id.append(-1)

            if sol.numOp < best_sol.numOp:
                best_sol = sol
                fixation_rate = 100
                time_increment_count = 0
                tt = time.time()
            else:
                count += 1

        elif solver.get_solve_status()=='Infeasible':
            print("Infeasible")
            count+=1
            fixation_rate-=fix_rate_decrement
            if len(flexible_tasks)==0:
                break
        elif solver.get_solve_status()=='Unknown':
            print("time exceed-")
            time_increment_count += 1
            fixation_rate -= fix_rate_decrement*1.5

        for i in range(len(flexible_tasks)):
            tasks[flexible_tasks[i]].set_start_max(TT_1)
            tasks[flexible_tasks[i]].set_start_min(0)
        #
        for i in range(len(cst)):
            model.remove(cst[i])
        model.remove(bound_cst)

    elapsed_secs = time.time() - tt
    # print("best_sol.op_profile-------------",best_sol.op_profile)
    # for i in range(nbTasks):
    #     print(best_sol.start_time[i],best_sol.end_time[i])
    if save_sol:
            solution_to_file(best_sol,filename,elapsed_secs)

    return best_sol

instances = np.load(f'biaozhun_16_12.npy', allow_pickle=True)
Init = InitM("dis.csv")
FixedMes.distance = Init.readDis()

cals = []
sols = []
values = []
if_optimal = []
times = [60,300,600]
for i in range(3):
    FixedMes.act_info = Init.readData(0, instances[0])
    sol  = rcpsp_solve(16,12,90,FixedMes.act_info, times[i], True,"LNS_16_11")

    print("目标值------------", sol)







