import numpy as np

from FixedMess import FixedMes
from Human import Human
from Space import Space
from Station import Station


def get_sort_index(lst):

    sorted_lst = sorted(lst,key=lambda x:x)
    sort_index = []
    for x in lst:
        indexo = sorted_lst.index(x)
        sort_index.append(indexo)
        sorted_lst[indexo] = 999999999

    return sort_index

def encoder(individual ,activities):

        sort_index = get_sort_index(individual)
        clone_index = copy.deepcopy(sort_index)
        numbers = len(activities)
        cloneA = copy.deepcopy(activities)
        unschedule = list(copy.deepcopy(individual))
        schedule = []
        while len(unschedule)>0:
            order_id = clone_index.index(min(sort_index))

            prece = cloneA[order_id].predecessor

            if len(prece)>0:
                while len(prece)>0:
                    indexs = [clone_index[i] for i in prece]
                    order_id = clone_index.index(min(indexs))
                    try:
                        prece = cloneA[order_id].predecessor
                    except:
                        print(order_id)
                        print("-------------")
            for key, Ei in cloneA.items():
                prece = cloneA[key].predecessor
                if order_id in prece:
                    prece.remove(order_id)
            schedule.append([order_id,cloneA[order_id].belong_plane_id,cloneA[order_id].taskid])
            del cloneA[order_id]
            del unschedule[unschedule.index(individual[order_id])]
            del sort_index[sort_index.index(clone_index[order_id])]
        return schedule

# TODO 编码是0-1的随机数，变成调度顺序编码

def Fitness(individual, activities,LR):

    list_order = encoder(individual, activities)
    acts,codes  = serialGenerationScheme(activities, list_order, LR)
    return acts[len(individual)-1].ef,np.array(codes)

def initMess():
        Humans = []
        Stations = []
        Spaces = []
        number = 0

        for i in range(FixedMes.Human_resource_type):
            Humans.append([])
            for j in range(FixedMes.total_Huamn_resource[i]):
                # ij都是从0开头 ,number也是

                Humans[i].append(Human([i,j,number]))
                number += 1

        number = 0

        for i in range(FixedMes.station_resource_type):
            Stations.append([])
            for j in range(len(FixedMes.constraintS_JZJ[i])):
                # ij都是从0开头 ,number也是
                Stations[i].append(Station([i,j,number]))
                number += 1

        for i in range(18):
            Spaces.append(Space(i))
        return Humans,Stations,Spaces

def parallelGenerationScheme(allTasks, codes,  LR):
    assert LR=='l'or LR=='r'
    if LR=='l':
        return parallelGenerationScheme_l(allTasks, codes)
    # else:
    #     allTasks,codes =parallelGenerationScheme_r(allTasks, codes)
    #     if allTasks[0].es>0:
    #         tmp = allTasks[0].es
    #         for i in range(len(list(allTasks.keys()))):
    #             allTasks[i].es-=tmp
    #             allTasks[i].ef-=tmp
    #     return allTasks,codes

def j_human(t,used,order):
    type_h =int(order.resourceHType)
    needh = order.needH
    for ti in range(int(t),int(t+order.duration)):
        if used[type_h][ti] + needh > FixedMes.total_Huamn_resource[type_h]:
            return False
    return True
def j_space(t, spaces, order):
    flag = True

    if order.Space > 0:
            flag = False
            now_pos = order.belong_plane_id
            space = spaces[now_pos - 1]

            if (len(space.OrderOver) == 0):
                flag = True # 该类资源可用+1
            else:
                Activity1 = space.OrderOver[-1]
                if (Activity1.ef) <= t:
                    flag = True  # 该类资源可用+1
    return flag
def j_station(t,stations,order):
    flag = True
    recordS = []
    type = int(order.RequestStationType)
    now_pos = order.belong_plane_id
    if type >= 0:
        flag = False
        for station in stations[int(type)]:
            # 舰载机在这个加油站的覆盖范围内：
            if now_pos in FixedMes.constraintS_JZJ[type][station.zunumber]:

                if (len(station.OrderOver) == 0):
                    flag = True  # 该类资源可用+1
                    recordS.append(station)

                else:

                    Activity1 = station.OrderOver[-1]
                    from_pos = Activity1.belong_plane_id
                    to_pos = Activity1.belong_plane_id
                    movetime1 = FixedMes.distance[from_pos][now_pos] * FixedMes.station_tranfer_speed
                    movetime2 = FixedMes.distance[now_pos][to_pos] * FixedMes.station_tranfer_speed

                    if (Activity1.ef) <= t:
                        flag = True  # 该类资源可用+1
                        recordS.append(station)
    return flag, recordS

def parallelGenerationScheme_l(allTasks, codes):
    humans, stations, spaces = initMess()
    humans_resources = [len(humans[i]) for i in range(len(humans))]
    used_humans = [[0] * 200 for i in range(4)]

    MTRCA(stations, codes, allTasks)
    num = len(codes)
    tmp_code = copy.deepcopy([c[0] for c in codes])
    stage = 0
    eligible = [0]
    scheduled = []
    t=0
    acting=[]
    finished = []
    start_time_codes = [ 0 for _ in range(len(codes))]
    while len(scheduled)<len(codes):

        if len(eligible)>0:
            eli = copy.deepcopy(eligible)
            for order_id in eli:
                if set(finished) >= set(allTasks[order_id].predecessor) and \
                        j_human(t, used_humans, allTasks[order_id]) and \
                        j_space(t, spaces, allTasks[order_id]):

                    f, r = j_station(t, stations, allTasks[order_id])
                    if f==False:
                        assert len(r)==0
                    if f==True:
                        pass
                    else:
                        eligible.remove(order_id)
                else:
                    eligible.remove(order_id)
        if len(eligible) == 0 and len(scheduled) < len(codes):
            ss = copy.deepcopy(acting)
            while len(eligible) == 0:

                ss = sorted(ss, key=lambda x: allTasks[x].ef)
                t = allTasks[ss[0]].ef

                for acting_order in ss:
                    if allTasks[acting_order].ef<=t:
                            finished.append(acting_order)
                            acting.remove(acting_order)
                ss = copy.deepcopy(acting)

                for order_id in tmp_code:
                    if set(finished) >= set(allTasks[order_id].predecessor) and \
                            j_human(t, used_humans, allTasks[order_id]) and \
                            j_space(t, spaces, allTasks[order_id]):

                        f, r = j_station(t, stations, allTasks[order_id])
                        if f==True:
                            eligible.append(order_id)


        order_id = eligible[0]

        tmp_code.remove(eligible[0])
        eligible.remove(eligible[0])
        scheduled.append(order_id)
        acting.append(order_id)

        dur = allTasks[order_id].duration
        now_pos = allTasks[order_id].belong_plane_id

        allTasks[order_id].es = round(t, 0)
        allTasks[order_id].ef = round(t + dur, 0)
        start_time_codes[order_id] = round(t, 0)

        # recordH1,humans1, allTasks, selectTaskID, now_pos
        type_human = allTasks[order_id].resourceHType
        needH = allTasks[order_id].needH
        for i in range(int(t), int(t+dur)):
                used_humans[type_human][i] += needH

        # recordS, stations, allTasks, selectTaskID, codes
        f, recordS = j_station(t, stations, allTasks[order_id])

        if len(recordS) > 0:
            allocationStation(recordS, stations, allTasks, order_id)

        need = allTasks[order_id].Space
        if need > 0:
            spaces[now_pos - 1].update(allTasks[order_id])

    #  codes[stage] = round(t, 0)
    return allTasks, start_time_codes
def serialGenerationScheme(allTasks, codes,  LR):
    assert LR=='l'or LR=='r'
    if LR=='l':
        return serialGenerationScheme_l(allTasks, codes)
    else:
        allTasks,codes =serialGenerationScheme_r(allTasks, codes)
        if allTasks[0].es>0:
            tmp = allTasks[0].es
            for i in range(len(list(allTasks.keys()))):
                allTasks[i].es-=tmp
                allTasks[i].ef-=tmp
        return allTasks,codes
def serialGenerationScheme_r(allTasks, codes):
    humans, stations, spaces = initMess()
    humans_resources = [len(humans[i]) for i in range(len(humans))]
    used_humans = [[0] * 200 for i in range(4)]
    codes= list(reversed(codes))
    MTRCA(stations, codes, allTasks)
    priorityToUse = sorted(allTasks.items(),key=lambda x:-x[1].ef)#最晚结束的最早开始

    end_time_codes = [0 for _ in range(len(codes))]

    for stage in range(0, len(priorityToUse)):
        selectTaskID = priorityToUse[stage][0]
        earliestEndTime = priorityToUse[0][1].ef

        '''
        需要考虑移动时间
        '''
        now_pos = allTasks[selectTaskID].belong_plane_id
        dur = allTasks[selectTaskID].duration
        for sucTaskID in allTasks[selectTaskID].successor:
            if allTasks[sucTaskID].es < earliestEndTime:
                earliestEndTime = allTasks[sucTaskID].es

        EndTime = earliestEndTime
        # 检查满足资源限量约束的时间点作为活动最早开始时间，即在这一时刻同时满足活动逻辑约束和资源限量约束
        t = EndTime
        recordH = []
        recordS = []

        type = allTasks[selectTaskID].resourceHType
        need = allTasks[selectTaskID].needH
        flag_human = False

        # 计算t时刻正在进行的活动的资源占用总量,当当前时刻大于活动开始时间小于等于活动结束时间时，说明活动在当前时刻占用资源
        while t <= EndTime:
            recordH = []
            recordS = []
            avil_human = 0

            if dur == 0:
                break
            # flag = judgeRenew(allTasks, stations, resourceSumNew, selectTaskID, t, dur)

            # 第舰载机的座舱资源
            flag_space = judgeSpace_r(allTasks, spaces, selectTaskID, now_pos, t, dur)
            flag_human = True

            for i in range(int(t-dur), int(t )):
                if used_humans[type][i] + need > humans_resources[type]:
                    flag_human = False
                    break


            #avil_human, recordH = judgeHuman_r(humans, type, need, now_pos, t, dur)

            stype = allTasks[selectTaskID].RequestStationType
            flag_station, recordS = judgeStation_r(stations, stype, now_pos, t, dur)

            # 若资源不够，则向后推一个单位时间
            if (flag_space == False) or (flag_human == False) or (flag_station == False):
                t = round(t - 1, 0)
            else:
                break
            # 若符合资源限量则将当前活动开始时间安排在这一时刻
        allTasks[selectTaskID].ef = round(t, 0)
        allTasks[selectTaskID].es = round(t - dur, 0)

        # recordH1,humans1, allTasks, selectTaskID, now_pos
        if flag_human == True:
            for i in range(int(t-dur), int(t)):
                used_humans[type][i] += need

        # recordS, stations, allTasks, selectTaskID, codes
        if len(recordS) > 0:
           allocationStation(recordS, stations, allTasks, selectTaskID)

        need = allTasks[selectTaskID].Space
        if need > 0:
            spaces[now_pos - 1].update(allTasks[selectTaskID])
      #  codes[stage] = round(t, 0)
        end_time_codes[selectTaskID] = round(t, 0)
    return allTasks,np.array(end_time_codes)
def judgeHuman_r(Human, type, need, now_pos, t, dur):
    # 全甲板模式
    if FixedMes.modeflag == 0:
        typeHuman = Human[type]
    else:
        typeHuman = Human[int((now_pos - 1) / FixedMes.modeflag)][type]
    recordH1 = [ ]
    avil_human = 0
    if need > 0:
        for human in typeHuman:
            if (len(human.OrderOver) == 0):
                avil_human += 1  # 该类资源可用+1
                recordH1.append(human)
                need-=1

            if (len(human.OrderOver) == 1):
                Activity1 = human.OrderOver[0]
                from_pos = Activity1.belong_plane_id
                to_pos = Activity1.belong_plane_id

                if (Activity1.ef ) <= (t-dur) \
                        or (t) <= (Activity1.es ):
                    avil_human += 1  # 该类资源可用+1
                    recordH1.append(human)
                    need -= 1

            # 遍历船员工序，找到可能可以插入的位置,如果船员没有工作，人力资源可用
            if (len(human.OrderOver) >= 2):
                flag = False
                for taskIndex in range(len(human.OrderOver) - 1,0,-1):
                    Activity1 = human.OrderOver[taskIndex]
                    Activity2 = human.OrderOver[taskIndex - 1]

                    from_pos = Activity2.belong_plane_id
                    to_pos = Activity1.belong_plane_id
                    movetime1 = FixedMes.distance[from_pos][now_pos] * FixedMes.human_walk_speed
                    movetime2 = FixedMes.distance[now_pos][to_pos] * FixedMes.human_walk_speed

                    if (Activity2.ef ) <= (t-dur) \
                            and (t) <= (Activity1.es ):
                        flag = True
                        avil_human += 1  # 该类资源可用+1
                        recordH1.append(human)
                        need -= 1
                        break
                if flag == False:
                    Activity1 = human.OrderOver[-1]
                    Activity2 = human.OrderOver[0]
                    from_pos = Activity2.belong_plane_id
                    to_pos = Activity1.belong_plane_id
                    movetime2 = FixedMes.distance[from_pos][now_pos] * FixedMes.human_walk_speed
                    movetime1 = FixedMes.distance[now_pos][to_pos] * FixedMes.human_walk_speed

                    if (Activity1.ef) <= (t-dur) \
                            or (t ) <= (Activity1.es):
                        avil_human += 1  # 该类资源可用+1
                        recordH1.append(human)
                        need -= 1

    return avil_human, recordH1
def judgeSpace_r(allTasks, spaces,  selectTaskID, now_pos, t, dur):
    flag = True
    if allTasks[selectTaskID].Space > 0:
            space = spaces[now_pos - 1]

            if (len(space.OrderOver) == 0):
                flag = True # 该类资源可用+1
            if (len(space.OrderOver) == 1):
                Activity1 = space.OrderOver[0]
                flag = False
                if (Activity1.ef) <= t-dur \
                        or (t) <= (Activity1.es):
                    flag = True  # 该类资源可用+1

            # 遍历，找到可能可以插入的位置,如果船员没有工作，人力资源可用
            if (len(space.OrderOver) >= 2):
                flag = False
                for taskIndex in range(len(space.OrderOver) - 1,0,-1):
                    Activity1 = space.OrderOver[taskIndex]
                    Activity2 = space.OrderOver[taskIndex - 1]
                    if (Activity2.ef) <= (t-dur) \
                            and (t ) <= (Activity1.es):
                        flag = True

                        break

                if flag == False:
                    Activity1 = space.OrderOver[-1]
                    Activity2 = space.OrderOver[0]

                    if (Activity1.ef) <= (t-dur) \
                            or (t) <= (Activity2.es):
                        flag = True  # 该类资源可用+1
    return flag
def judgeStation_r( Station, type, now_pos, t, dur):
        flag = True
        recordS = []
        if type >= 0:
            for station in Station[int(type)]:
                # 舰载机在这个加油站的覆盖范围内：
                if now_pos in FixedMes.constraintS_JZJ[type][station.zunumber]:

                    if (len(station.OrderOver) == 0):
                        flag = True  # 该类资源可用+1
                        recordS.append(station)

                    if (len(station.OrderOver) == 1):
                        flag = False
                        Activity1 = station.OrderOver[0]
                        from_pos = Activity1.belong_plane_id
                        to_pos = Activity1.belong_plane_id
                        movetime1 = FixedMes.distance[from_pos][now_pos] * FixedMes.station_tranfer_speed
                        movetime2 = FixedMes.distance[now_pos][to_pos] * FixedMes.station_tranfer_speed

                        if (Activity1.ef ) <= (t-dur) \
                                or (t) <= (Activity1.es):
                            flag = True  # 该类资源可用+1
                            recordS.append(station)

                    if (len(station.OrderOver) >= 2):
                        flag = False
                        for taskIndex in range(len(station.OrderOver) - 1,0,-1):
                            Activity1 = station.OrderOver[taskIndex]
                            Activity2 = station.OrderOver[taskIndex - 1]

                            from_pos = Activity1.belong_plane_id
                            to_pos = Activity2.belong_plane_id
                            movetime1 = FixedMes.distance[from_pos][now_pos] * FixedMes.station_tranfer_speed
                            movetime2 = FixedMes.distance[now_pos][to_pos] * FixedMes.station_tranfer_speed

                            if (Activity2.ef ) <= t-dur \
                                    and (t) <= (Activity1.es ):
                                flag = True  # 该类资源可用+1
                                recordS.append(station)
                                #flag = True
                        if flag == False:
                            Activity1 = station.OrderOver[-1]
                            Activity2 = station.OrderOver[0]

                            from_pos = Activity1.belong_plane_id
                            to_pos = Activity2.belong_plane_id
                            movetime1 = FixedMes.distance[from_pos][now_pos] * FixedMes.station_tranfer_speed
                            movetime2 = FixedMes.distance[now_pos][to_pos] * FixedMes.station_tranfer_speed

                            if (Activity1.ef ) <= (t-dur) or (t) <= (Activity1.es):
                                flag = True
                                recordS.append(station)
        return flag, recordS
def serialGenerationScheme_l(allTasks, codes):
    humans, stations, spaces = initMess()
    humans_resources = [len(humans[i]) for i in range(len(humans))]
    used_humans = [[0]*200 for i in range(4)]
    MTRCA(stations, codes, allTasks)

    # 记录资源转移
    priorityToUse = codes
    ps = [0]  # 局部调度计划初始化

    allTasks[0].es = 0  # 活动1的最早开始时间设为0
    allTasks[0].ef = allTasks[0].es + allTasks[0].duration

    start_time_codes = [0 for _ in range(len(codes))]
    for stage in range(0, len(priorityToUse)):
        selectTaskID = priorityToUse[stage][0]
        earliestStartTime = 0

        '''
        需要考虑移动时间
        '''
        now_pos = allTasks[selectTaskID].belong_plane_id
        dur = allTasks[selectTaskID].duration
        for preTaskID in allTasks[selectTaskID].predecessor:
            if allTasks[preTaskID].ef > earliestStartTime:
                earliestStartTime = allTasks[preTaskID].ef

        startTime = earliestStartTime
        # 检查满足资源限量约束的时间点作为活动最早开始时间，即在这一时刻同时满足活动逻辑约束和资源限量约束
        t = startTime
        recordH = []
        recordS = []
        flag_human = False
        type = allTasks[selectTaskID].resourceHType
        need = allTasks[selectTaskID].needH

        # 计算t时刻正在进行的活动的资源占用总量,当当前时刻大于活动开始时间小于等于活动结束时间时，说明活动在当前时刻占用资源
        while t >= startTime:
            recordH = []
            recordS = []
            avil_human = 0

            if dur == 0:
                break
            # flag = judgeRenew(allTasks, stations, resourceSumNew, selectTaskID, t, dur)

            # 第舰载机的座舱资源
            flag_space = judgeSpace_l(allTasks, spaces, selectTaskID, now_pos, t, dur)

            #avil_human, recordH = judgeHuman_l(humans, type, need, now_pos, t, dur)
            flag_human = True
            for i in range(int(t), int(t+dur)):
                if used_humans[type][i] + need > humans_resources[type]:
                    flag_human = False
                    break

            s_type = allTasks[selectTaskID].RequestStationType
            flag_station, recordS = judgeStation_l(stations, s_type, now_pos, t, dur)

            # 若资源不够，则向后推一个单位时间
            if (flag_space == False) or (flag_human == False) or (flag_station == False):
                t = round(t + 1, 0)
            else:
                break
            # 若符合资源限量则将当前活动开始时间安排在这一时刻
        allTasks[selectTaskID].es = round(t, 0)
        allTasks[selectTaskID].ef = round(t + dur, 0)

        # recordH1,humans1, allTasks, selectTaskID, now_pos
        if flag_human == True:
            for i in range(int(t),int(t+dur)):
                used_humans[type][i] += need

            #allocationHuman(recordH, humans, allTasks, selectTaskID, now_pos)

        # recordS, stations, allTasks, selectTaskID, codes
        if len(recordS) > 0:
            allocationStation(recordS, stations, allTasks, selectTaskID)

        need = allTasks[selectTaskID].Space
        if need > 0:
            spaces[now_pos - 1].update(allTasks[selectTaskID])

        # 局部调度计划ps
        ps.append(selectTaskID)
        start_time_codes[selectTaskID] = round(t, 0)
    return allTasks,np.array(start_time_codes)
def judgeHuman(Human, type, need, now_pos, t):
    # 全甲板模式
    if FixedMes.modeflag == 0:
        typeHuman = Human[type]
    else:
        typeHuman = Human[int((now_pos - 1) / FixedMes.modeflag)][type]
    recordH1 = [ ]
    avil_human = 0
    if need > 0:
        for human in typeHuman:
            if (len(human.OrderOver) == 0):
                avil_human += 1  # 该类资源可用+1
                recordH1.append(human)
                need-=1

            else:
                Activity1 = human.OrderOver[-1]
                from_pos = Activity1.belong_plane_id
                to_pos = Activity1.belong_plane_id
                movetime1 = FixedMes.distance[from_pos][now_pos] * FixedMes.human_walk_speed

                movetime2 = FixedMes.distance[now_pos][to_pos] * FixedMes.human_walk_speed

                if (Activity1.ef ) <= t:
                    avil_human += 1  # 该类资源可用+1
                    recordH1.append(human)
                    need -= 1
    return avil_human, recordH1
def judgeSpace_l(allTasks, spaces,  selectTaskID, now_pos, t, dur):
    flag = True
    if allTasks[selectTaskID].Space > 0:
            space = spaces[now_pos - 1]

            if (len(space.OrderOver) == 0):
                flag = True # 该类资源可用+1
            if (len(space.OrderOver) == 1):
                Activity1 = space.OrderOver[0]
                flag = False
                if (Activity1.ef) <= t \
                        or (t + dur) <= (Activity1.es):
                    flag = True  # 该类资源可用+1


            # 遍历船员工序，找到可能可以插入的位置,如果船员没有工作，人力资源可用
            if (len(space.OrderOver) >= 2):
                flag = False
                for taskIndex in range(len(space.OrderOver) - 1):
                    Activity1 = space.OrderOver[taskIndex]
                    Activity2 = space.OrderOver[taskIndex + 1]
                    if (Activity1.ef) <= t \
                            and (t + dur) <= (Activity2.es):
                        flag = True

                        break

                if flag == False:
                    Activity1 = space.OrderOver[0]
                    Activity2 = space.OrderOver[-1]

                    if (Activity2.ef) <= t \
                            or (t + dur) <= (Activity1.es):
                        flag = True  # 该类资源可用+1
    return flag
def judgeStation_l( Station, type, now_pos, t, dur):
        flag = True
        recordS = []
        if type >= 0:
            for station in Station[int(type)]:
                # 舰载机在这个加油站的覆盖范围内：
                if now_pos in FixedMes.constraintS_JZJ[type][station.zunumber]:

                    if (len(station.OrderOver) == 0):
                        flag = True  # 该类资源可用+1
                        recordS.append(station)

                    if (len(station.OrderOver) == 1):

                        Activity1 = station.OrderOver[0]
                        from_pos = Activity1.belong_plane_id
                        to_pos = Activity1.belong_plane_id
                        movetime1 = FixedMes.distance[from_pos][now_pos] * FixedMes.station_tranfer_speed
                        movetime2 = FixedMes.distance[now_pos][to_pos] * FixedMes.station_tranfer_speed

                        if (Activity1.ef ) <= t \
                                or (t + dur) <= (Activity1.es):
                            flag = True  # 该类资源可用+1
                            recordS.append(station)

                    if (len(station.OrderOver) >= 2):
                        ff=False
                        for taskIndex in range(len(station.OrderOver) - 1):
                            Activity1 = station.OrderOver[taskIndex]
                            Activity2 = station.OrderOver[taskIndex + 1]

                            from_pos = Activity1.belong_plane_id
                            to_pos = Activity2.belong_plane_id
                            movetime1 = FixedMes.distance[from_pos][now_pos] * FixedMes.station_tranfer_speed
                            movetime2 = FixedMes.distance[now_pos][to_pos] * FixedMes.station_tranfer_speed

                            if (Activity1.ef ) <= t \
                                    and (t + dur ) <= (Activity2.es ):
                                flag = True  # 该类资源可用+1
                                ff = True
                                recordS.append(station)
                                #flag = True
                        if ff == False:
                            Activity1 = station.OrderOver[-1]
                            Activity2 = station.OrderOver[0]

                            from_pos = Activity1.belong_plane_id
                            to_pos = Activity2.belong_plane_id
                            movetime1 = FixedMes.distance[from_pos][now_pos] * FixedMes.station_tranfer_speed
                            movetime2 = FixedMes.distance[now_pos][to_pos] * FixedMes.station_tranfer_speed

                            if (Activity1.ef ) <= t or (t + dur) <= (Activity2.es):
                                flag = True
                                recordS.append(station)
        return flag, recordS
import copy
def allocationHuman(recordH1,humans1, allTasks, selectTaskID, now_pos):
            type = allTasks[selectTaskID].resourceHType
            need = allTasks[selectTaskID].needH
            r = copy.deepcopy(recordH1)
            huamn_ids = []
            if need==0:
                huamn_ids.append((-1,-1))
                return huamn_ids
            while need > 0:

                rr = sorted(r, key=lambda x:x.alreadyworkTime)
                choose_human = rr[0]
                index = choose_human.zunumber
                r.remove(choose_human)

                # 更新人员
                if FixedMes.modeflag==0:
                    humans1[type][index].update(allTasks[selectTaskID])
                    allTasks[selectTaskID].HumanNums.append([type, index])
                else:
                    humans1[int((now_pos-1)/FixedMes.modeflag)][type][index].update(allTasks[selectTaskID])
                    allTasks[selectTaskID].HumanNums.append([type, index])

                # allTasks[selectTaskID].HumanNums.append(humans[type][index].number)
                need -= 1
                huamn_ids.append((type,index))
            return huamn_ids

def dfs_se(used_humans,allTasks, codes,ID,humans, stations, spaces,SB):
    if ID == len(codes):
        if allTasks[ID-1].ef < FixedMes.MAXT:
            FixedMes.MAXT = allTasks[ID-1].ef
            FixedMes.SheBei = copy.deepcopy(SB)
            FixedMes.start_time_codes = [allTasks[i].es for i in range(len(codes))]
        return allTasks[ID-1].ef

    earliestStartTime = 0
    humans_resources = [len(humans[i]) for i in range(len(humans))]
    selectID = codes[ID][0]
    now_pos = allTasks[selectID].belong_plane_id
    dur = allTasks[selectID].duration
    for preTaskID in allTasks[selectID].predecessor:
        if allTasks[preTaskID].ef > earliestStartTime:
            earliestStartTime = allTasks[preTaskID].ef

    startTime = earliestStartTime
    # 检查满足资源限量约束的时间点作为活动最早开始时间，即在这一时刻同时满足活动逻辑约束和资源限量约束

    type = allTasks[selectID].resourceHType
    need = allTasks[selectID].needH

    stype = allTasks[selectID].RequestStationType
    now_pos = allTasks[selectID].belong_plane_id

    needspace = allTasks[selectID].Space
    if stype < 0:
        t = startTime
        flag_human = False
        while t >= startTime:
            if dur == 0:
                break

            # 第舰载机的座舱资源
            flag_space = judgeSpace_l(allTasks, spaces, selectID, now_pos, t, dur)
            flag_human = True
            for i in range(int(t), int(t + dur)):
                if used_humans[type][i] + need > humans_resources[type]:
                    flag_human = False
                    break
            flag_station = True

            # 若资源不够，则向后推一个单位时间
            if (flag_space == False) or (flag_human == False) or (flag_station == False):
                t = round(t + 1, 0)
            else:
                break
            # 若符合资源限量则将当前活动开始时间安排在这一时刻
        allTasks[selectID].es = round(t, 0)
        allTasks[selectID].ef = round(t + dur, 0)
        # recordH1,humans1, allTasks, selectTaskID, now_pos
        if flag_human == True:
            for i in range(int(t), int(t + dur)):
                used_humans[type][i] += need
        need = allTasks[selectID].Space
        if need > 0:
            spaces[now_pos - 1].update(allTasks[selectID])
        dfs_se(used_humans, allTasks, codes, ID + 1, humans, stations, spaces, SB)

    if stype >= 0:
        for station in stations[int(stype)]:
            # 舰载机在这个加油站的覆盖范围内：
            if now_pos in FixedMes.constraintS_JZJ[stype][station.zunumber]:
                t = startTime
                flag_human = False
                while t >= startTime:

                    if dur == 0:
                        break
                    # 第舰载机的座舱资源
                    flag_space = judgeSpace_l(allTasks, spaces, selectID, now_pos, t, dur)

                    # avil_human, recordH = judgeHuman_l(humans, type, need, now_pos, t, dur)
                    flag_human = True
                    for i in range(int(t), int(t + dur)):
                        if used_humans[type][i] + need > humans_resources[type]:
                            flag_human = False
                            break

                    s_type = int(allTasks[selectID].RequestStationType)
                    nu = int(station.zunumber)
                    flag_station = True
                    station = stations[s_type][nu]
                    if len(station.OrderOver) > 0:
                        if len(station.OrderOver) == 1:
                            if t >= station.OrderOver[0].ef:
                                pass
                            elif t + dur <= station.OrderOver[0].es:
                                pass
                            else:
                                flag_station = False
                        else:
                            assert len(station.OrderOver) > 1
                            order = station.OrderOver[0]
                            flag_station = False
                            if order.es >= (t + dur):
                                flag_station = True
                                break
                            for order_id in range(1, len(station.OrderOver)):
                                if station.OrderOver[order_id].es >= t + dur and t >= station.OrderOver[
                                    order_id - 1].ef:
                                    flag_station = True
                                    break
                            if t >= station.OrderOver[-1].es:
                                flag_station = True

                    # 若资源不够，则向后推一个单位时间
                    if (flag_space == False) or (flag_human == False) or (flag_station == False):
                        t = round(t + 1, 0)
                    else:
                        break
                    # 若符合资源限量则将当前活动开始时间安排在这一时刻
                allTasks[selectID].es = round(t, 0)
                allTasks[selectID].ef = round(t + dur, 0)

                if flag_human == True:
                    for i in range(int(t), int(t + dur)):
                        used_humans[type][i] += need
                station.update(allTasks[selectID])
                if needspace > 0:
                    spaces[now_pos - 1].update(allTasks[selectID])
                SB[selectID] = [stype,station.zunumber]
                dfs_se(used_humans, allTasks, codes, ID+1, humans, stations, spaces, SB)


                if flag_human == True:
                    for i in range(int(t), int(t + dur)):
                        used_humans[type][i] -= need
                station.delete(allTasks[selectID])
                if needspace > 0:
                    spaces[now_pos - 1].delete(allTasks[selectID])

def dfs_serialGenerationScheme(allTasks, codes):
    humans, stations, spaces = initMess()

    used_humans = [[0]*2000000 for _ in range(4)]
    MTRCA(stations, codes, allTasks)

    # 记录资源转移
    priorityToUse = codes
    ps = [0]  # 局部调度计划初始化
    start_time_codes = [0 for _ in range(len(codes))]
    allTasks[0].es = 0  # 活动1的最早开始时间设为0
    allTasks[0].ef = allTasks[0].es + allTasks[0].duration
    FixedMes.SheiBei = [[] for _ in range(len(codes))]
    FixedMes.MAXT = 99999
    FixedMes.start_time_codes = [0 for _ in range(len(codes))]
    SB = [[] for _ in range(len(codes))]
    dfs_se(used_humans, allTasks, codes, 0, humans, stations, spaces, SB)

    return allTasks,np.array(start_time_codes)
def allocationStation(recordS, stations, allTasks, selectTaskID):
        jzj = allTasks[selectTaskID].belong_plane_id
        t = allTasks[selectTaskID].es
        dur = allTasks[selectTaskID].duration
        # 基于规则
        record = {}
        for station in recordS:
            if len(station.OrderOver) == 0:
                record[station.zunumber]=0
            elif len(station.OrderOver) == 1:
                if t >= station.OrderOver[0].ef:
                    record[station.zunumber]=0
                else:
                    assert t + dur <= station.OrderOver[0].es
                    record[station.zunumber] = station.OrderOver[0].es-t-dur
            else:
                assert len(station.OrderOver) > 1
                order = station.OrderOver[0]
                if order.es >= (t + dur):
                    record[station.zunumber] =  order.es-t-dur
                    continue
                for order_id in range(1, len(station.OrderOver)):
                    if station.OrderOver[order_id].es >= t + dur:
                        record[station.zunumber] = order.es - t - dur
                        break

                if t >=station.OrderOver[-1].es:
                    record[station.zunumber]= 0
        choose_s = sorted(recordS,key=lambda x:(x.fugai_time,record[x.zunumber]))[0]
        type = choose_s.type
        nu = choose_s.zunumber
        # 更新
        stations[type][nu].update(allTasks[selectTaskID])
        allTasks[selectTaskID].SheiBei.append([type, nu])

        for station in stations[type]:
            index = station.zunumber
            if jzj in FixedMes.constraintS_JZJ[type][index]:
                station.fugai_time -=allTasks[selectTaskID].duration
        return nu

def MTRCA(Stations, nodes, alltasks):

    for next in nodes:
        id = next[0]
        jzj = next[1]
        if alltasks[id].RequestStationType >=0:
            type = int(alltasks[id].RequestStationType)
            for station in Stations[type]:
                index = station.zunumber
                if jzj in FixedMes.constraintS_JZJ[type][index]:
                    station.fugai_time += alltasks[id].duration



