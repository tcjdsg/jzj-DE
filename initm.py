#读取数据
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from FixedMess import FixedMes
from activity import Order


class InitM(object):
    def __init__(self, filenameDis):

        self.filename1 = filenameDis


        self.humanMoveTime = []
        self.num_activities = 0
        self.num_resource_type = 0
        self.total_resource = []
        self.activities = { }
    def readDis(self):

        dis = pd.read_csv(self.filename1, header=None,encoding="utf-8").values
        pdis = dis.tolist()
        for i in range(1,dis.shape[0]):
            for j in range(1,dis.shape[1]):
                pdis[i][j] = round(int(dis[i][j]) * 1  , 1 )#单位是m
                #不考虑设备接口转移速度
        return pdis[1:][1:]

    def readData(self,r,insatnce):
        FixedMes.total_Huamn_resource =[int(n) for n in insatnce[2]]
        fea = insatnce[1]
        adj = insatnce[0]
        '''
        包括  1.活动数   2.项目资源数 3.项目资源种类数   4.项目资源限量
        5.所有活动的ID，持续时间，资源需求，紧前活动
        :param fileName:
        :return: activities:这个是标准单机流程
        '''
        # f = open(self.filename2)
        # jzjnums = f.readline().split(' ')
        # jzjNumbers = [ int(jzjnums[i]) for i in range(len(jzjnums))]
        # taskAndResourceType = f.readline().split(' ')  # 第一行数据包含活动数和资源数
        # num_activities = int(taskAndResourceType[0])  # 得到活动数
        # num_resource_type = int(taskAndResourceType[1])  # 得到资源类数
        # total_resource = np.array([int(value) for value in f.readline().split(' ')[:]])  # 获取资源限量
        # 将每个活动的所有信息存入到对应的Activity对象中去
        activities = {}
        index = -1
        # 构建任务网络
        dur = []
        for i in range(len(fea)):
                index += 1
                taskId = i
                pos_id = int(fea[index][-2])
                duration = fea[index][0]

                # nt()pri
                resourceH_type = sum(fea[index][1:5])
                need = 0
                if resourceH_type==0:
                    resourceH_type = -1
                else:
                    resourceH_type = np.where(fea[index][1:5]>0)[0][0]
                    need = fea[index][resourceH_type + 1]

                resourceS_type = fea[index][-1]-1
                if resourceS_type <0:
                    resourceS_type = -1

                resourceSpace = fea[index][-3]
                if duration == 0:
                    need = 0
                    resourceH_type = -1
                    resourceS_type = -1
                    resourceSpace = 0

                S = np.where(adj[index] > 0)[0]

                SUCOrder = []
                for o in S:
                    if o!=i:
                        SUCOrder.append(o)

                PreOrder = []
                P = np.where(adj[:,index].reshape(-1) > 0)[0]
                for p in P :
                    if p!=i:
                        PreOrder.append(p)

                task = Order(index, taskId, duration, resourceH_type, need, resourceS_type, resourceSpace, SUCOrder, pos_id)
                task.predecessor = PreOrder
                #task.vacp = vacp
                activities[index] = task

        return  activities
        # 活动数int， 资源数int， 资源限量np.array， 所有活动集合dic{活动代号：活动对象}


if __name__ == '__main__':
    m = InitM("D:/JZJ_GA/dis.csv")
    m.readDis()
    instanc = np.load(f'D:/JZJ_GA/JZJ_dataset/problems_4.npy', allow_pickle=True)[0]
    m.readData(0,instanc)
    print()












