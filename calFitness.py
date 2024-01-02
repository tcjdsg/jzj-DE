from FixedMess import FixedMes
from Human import Human
from Space import Space
from Station import Station


def get_sort_index(lst):

    sorted_lst = sorted(lst)
    sort_index = [sorted_lst.index(x) for x in lst]
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
                    prece = cloneA[order_id].predecessor
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
    Humans = []
    Stations = []
    Spaces = []
    initMess(Humans, Stations, Spaces)
    list_order = encoder(individual, activities)
    acts  = serialGenerationScheme(activities, list_order, Humans, Stations, Spaces, LR)
    return acts[len(individual)-1].ef

def initMess(Humans,Stations,Spaces):
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
            for j in range(FixedMes.total_station_resource[i]):
                # ij都是从0开头 ,number也是
                Stations[i].append(Station([i,j,number]))
                number += 1

        for i in range(18):
            Spaces.append(Space(i))

def serialGenerationScheme(allTasks, codes, humans, stations, spaces, LR):
    MTRCA(stations, codes, allTasks)

    # 记录资源转移
    priorityToUse = codes
    ps = [0]  # 局部调度计划初始化

    allTasks[0].es = 0  # 活动1的最早开始时间设为0
    allTasks[0].ef = allTasks[0].es + allTasks[0].duration

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

        # 计算t时刻正在进行的活动的资源占用总量,当当前时刻大于活动开始时间小于等于活动结束时间时，说明活动在当前时刻占用资源
        while t >= startTime:
            recordH = []
            recordS = []
            avil_human = 0

            if dur == 0:
                break
            # flag = judgeRenew(allTasks, stations, resourceSumNew, selectTaskID, t, dur)

            # 第舰载机的座舱资源
            flag_space = judgeSpace(allTasks, spaces, selectTaskID, now_pos, t, dur)

            type = allTasks[selectTaskID].resourceHType
            need = allTasks[selectTaskID].needH
            avil_human, recordH = judgeHuman(humans, type, need, now_pos, t, dur)

            type = allTasks[selectTaskID].RequestStationType
            flag_station, recordS = judgeStation(stations, type, now_pos, t, dur)

            # 若资源不够，则向后推一个单位时间
            if (flag_space == False) or (avil_human < need) or (flag_station == False):
                t = round(t + 1, 0)
            else:
                break
            # 若符合资源限量则将当前活动开始时间安排在这一时刻
        allTasks[selectTaskID].es = round(t, 0)
        allTasks[selectTaskID].ef = round(t + dur, 0)

        # recordH1,humans1, allTasks, selectTaskID, now_pos
        if len(recordH) > 0:
            allocationHuman(recordH, humans, allTasks, selectTaskID, now_pos)

        # recordS, stations, allTasks, selectTaskID, codes
        if len(recordS) > 0:
            allocationStation(recordS, stations, allTasks, selectTaskID)

        need = allTasks[selectTaskID].Space
        if need > 0:
            spaces[now_pos - 1].update(allTasks[selectTaskID])

        # 局部调度计划ps
        ps.append(selectTaskID)
    return allTasks
def judgeHuman(Human, type, need, now_pos, t, dur):
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
                movetime1 = FixedMes.distance[from_pos][now_pos] * FixedMes.human_walk_speed
                try:
                   movetime2 = FixedMes.distance[now_pos][to_pos] * FixedMes.human_walk_speed
                except:
                    print()
                if (Activity1.ef ) <= t \
                        or (t + dur) <= (Activity1.es ):
                    avil_human += 1  # 该类资源可用+1
                    recordH1.append(human)
                    need -= 1

            # 遍历船员工序，找到可能可以插入的位置,如果船员没有工作，人力资源可用
            if (len(human.OrderOver) >= 2):
                flag = False
                for taskIndex in range(len(human.OrderOver) - 1):
                    Activity1 = human.OrderOver[taskIndex]
                    Activity2 = human.OrderOver[taskIndex + 1]

                    from_pos = Activity1.belong_plane_id
                    to_pos = Activity2.belong_plane_id
                    movetime1 = FixedMes.distance[from_pos][now_pos] * FixedMes.human_walk_speed
                    movetime2 = FixedMes.distance[now_pos][to_pos] * FixedMes.human_walk_speed

                    if (Activity1.ef ) <= t \
                            and (t + dur) <= (Activity2.es ):
                        flag = True
                        avil_human += 1  # 该类资源可用+1
                        recordH1.append(human)
                        need -= 1
                        break
                if flag == False:
                    Activity1 = human.OrderOver[0]
                    Activity2 = human.OrderOver[-1]
                    from_pos = Activity2.belong_plane_id
                    to_pos = Activity1.belong_plane_id
                    movetime2 = FixedMes.distance[from_pos][now_pos] * FixedMes.human_walk_speed
                    movetime1 = FixedMes.distance[now_pos][to_pos] * FixedMes.human_walk_speed

                    if (Activity2.ef) <= t \
                            or (t + dur) <= (Activity1.es):
                        avil_human += 1  # 该类资源可用+1
                        recordH1.append(human)
                        need -= 1

    return avil_human, recordH1

def judgeSpace(allTasks, spaces,  selectTaskID, now_pos, t, dur):
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
def judgeStation( Station, type, now_pos, t, dur):
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

                        if (Activity1.ef ) <= t \
                                or (t + dur) <= (Activity1.es):
                            flag = True  # 该类资源可用+1
                            recordS.append(station)

                    if (len(station.OrderOver) >= 2):
                        flag = False
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
                                recordS.append(station)
                                #flag = True
                        if flag == False:
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


def allocationStation(recordS, stations, allTasks, selectTaskID):
        jzj = allTasks[selectTaskID].belong_plane_id
        # 基于规则
        choose_s = sorted(recordS,key=lambda x:x.fugai_time)[0]
        type = choose_s.type
        nu = choose_s.zunumber
        # 更新
        stations[type][nu].update(allTasks[selectTaskID])
        allTasks[selectTaskID].SheiBei.append([type, nu])

        for station in stations[type]:
            index = station.zunumber
            if jzj in FixedMes.constraintS_JZJ[type][index]:
                station.fugai_time -=allTasks[selectTaskID].duration

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