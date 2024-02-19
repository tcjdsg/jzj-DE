import random
from collections import defaultdict


class FixedMes(object):
    SheBei = []
    MAXT = 9999
    start_time_codes=[]
    """
    distance:
    orderInputMes:
    """

    jzj = [
           [0, 0, 4, 8, 3, 6, 5, 4, 0, 12, 6, 3, 4, 12, 8, 3, 10, 10, 7, 0],
           [0, 0, 4, 8, 4, 6, 6, 4, 5, 12, 0, 6, 4, 12, 8, 0, 10, 10, 7, 0],
           [0, 0, 6, 8, 5, 6, 0, 4, 0, 14, 6, 3, 4, 15, 0, 5, 0, 0, 12, 0],
           [0, 0, 4, 8, 3, 6, 5, 4, 6, 14, 0, 0, 4, 12, 8, 5, 12, 12, 7, 0],

           [0, 0, 4, 8, 3, 6, 5, 4, 6, 14, 6, 3, 4, 12, 8, 5, 12, 12, 7, 0],
           [0, 0, 3, 6, 4, 6, 3, 3, 6, 11, 4, 3, 4, 12, 8, 0, 8, 8, 12, 0],
           [0, 0, 3, 6, 5, 6, 3, 3, 8, 11, 5, 3, 3, 12, 8, 3, 9, 9, 12, 0],
           [0, 0, 4, 8, 4, 6, 0, 6, 0, 18, 10, 5, 3, 12, 8, 5, 0, 0, 10, 0],

           [0, 0, 3, 6, 4, 5, 5, 3, 9, 12, 4, 3, 4, 12, 8, 5, 8, 8, 8, 0],
           [0, 0, 5, 9, 5, 7, 4, 3, 8, 12, 4, 5, 4, 12, 8, 3, 8, 6, 8, 0],
           [0, 0, 4, 7, 5, 6, 8, 5, 8, 15, 4, 5, 3, 12, 8, 3, 12, 12, 6, 0],
           [0, 0, 3, 6, 3, 5, 5, 5, 6, 12, 4, 3, 3, 12, 8, 5, 10, 8, 7, 0]
           ]



    distance = [[]]

    numJzjPos = 18
    numHumanAll = [18,60]

    planeOrderNum = 12
    planeNum = 18
    jzjNumbers=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16]  #舰载机编号

    RU_time = [1,2,3,4,5,6,7,8]

    #座舱限制。相当于是每个站位都有一个座舱，每个舰载机只能用自己座舱。
    space_resource_type = planeNum
    total_space_resource = [1 for i in range(planeNum)]

    Human_resource_type = 4 #先考虑只有一类人
    # 特设、航电、军械、机械
  # 每种人员数量

    total_Huamn_resource = [30]
    constraintOrder = defaultdict(lambda: []) #记录每类人的可作用工序，和可作用舰载机范围
    # constraintOrder[0] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    constraintOrder[0] = [1, 2, 5]
    constraintOrder[1] = [3,4, 16,17]
    constraintOrder[2] = [7, 8, 14,16]
    constraintOrder[3] = [6, 9, 10, 11,12,13,15]

    modeflag = 0 #0是单机、1是全甲板，这里考虑全甲板，如果是全甲板
    # constraintJZJ = defaultdict(lambda: []) #保障人员可作用舰载机范围,两种模式，单机或者全甲板

    station_resource_type = 5
    total_station_resource = [6, 12, 6, 5, 6]

    # 飞机数量比较少的时候，这些燃料资源的限制约束不起作用。
    # total_renew_resource = [5,5,2,4,2]
    # # total_renew_resource = [1,1,1,1,1]
    total_renew_resource = [99, 99, 99, 99, 99]

    # 设备保障范围约束
    constraintS_JZJ = defaultdict(lambda: [])
   #加油、供电、充氧、充氮
    constraintS_JZJ[0] = [[1, 2, 3],
                          [3, 4, 5],
                          [6, 7],
                          [7, 8, 9],
                          [9, 10, 11],
                          [11,12,13,14],
                          [14,15,16]]

    constraintS_JZJ[1] = [[1],
                          [2],
                          [3],
                          [4],
                          [5],
                          [6],
                          [7],
                          [8],
                          [9],
                          [10],
                          [11],
                          [12],
                          [13],
                          [14],
                          [15],
                          [16]]

    constraintS_JZJ[2] = [[1, 2, 3, 4],
                          [4, 5, 6],
                          [7,8,9],
                          [9,10,11],
                          [12, 13, 14],
                          [14, 15,16]
                          ]

    constraintS_JZJ[3] = [[1, 2, 3, 4],
                          [4, 5, 6],
                          [7, 8, 9],
                          [9, 10, 11],
                          [12, 13,14],
                          [14,15,16]
                          ]

    constraintS_JZJ[4] = [[1, 2, 3],
                          [2, 3, 4],
                          [4, 5, 6],
                          [7, 8, 9],
                          [9, 10],
                          [11, 12],
                          [12, 13, 14],
                          [14, 15, 16]
                          ]

    Activity_num  = (planeOrderNum)*planeNum+2 #活动数量

    BestFit = {}

    #17位 为了让虚拟从1开始
    sigma = 0.3
    shedule_num=0
    act_info={}

    cross = 0.8
    cross1 = 2.5
    MutationRate = 0.3
    MutationRatePmo = 0.1

    transferrate = 0.2
    transfer_iter = 50
    human_walk_speed = 0# 0.75 #人员行走速度0.75/m
    station_tranfer_speed = 0 # 0.45

    populationnumber = 30
    ge = 60
    # 记录每一代的最优染色体
    threadNum = 1

    AgenarationIten = ge / 3
    GenarationIten = 0
    #保存每代染色体信息 父代
    AllFit = []
    AllFitSon = []
    AllFitFamily = []

    resver_k1 = [ 0 for _ in range(ge)]
    resver_k2 = [ 0 for _ in range(ge)]
    #populationnumber*populationnumber
    slect_F_step_alone = [[] for _ in range(populationnumber)]
    # slect_F_step = [[] for _ in range(populationnumber)]

    Paternal = [[0,0] for _ in range(int(populationnumber/2))]
    #每一代的平均值
    Avufit = {}
    BestCmax = {}
    BestPr = {}
    BestEcmax = {}
    Bestzonghe = {}
    var ={}
    f = {}
    d = {}
    m = {}

    AverPopmove = 0
    AverPopTime = 0
    AverPopVar = 0
    Diversity = 0.0
    keyChainOrder = []
    #死锁辅助检查列表
    # dealLockList=[[0 for _ in range(Activity_num)] for _ in range(Activity_num)]

    bestHumanNumberTarget=[]

    Allactivity = []
    constraintHuman =[]
    constraintStation=[]
    constraintSpace = []

    humanNum = 0
    targetWeight =[1,0.3,0.1]
    boundUpper =[0,0]
    boundLowwer=[]
    AON = []

    # 工序顺序
    SUCOrder = defaultdict(lambda: [])
    SUCOrder[0] = [1, 2]
    SUCOrder[1] = [3]
    SUCOrder[2] = [3]
    SUCOrder[3] = [4, 5, 6, 7]
    SUCOrder[4] = [12]
    SUCOrder[5] = [8, 9]
    SUCOrder[6] = [9]
    SUCOrder[7] = [10]
    SUCOrder[8] = [11]
    SUCOrder[9] = [12]
    SUCOrder[10] = [12]
    SUCOrder[11] = [12]
    SUCOrder[12] = [13]
    SUCOrder[13] = []
    #
    # SUCOrder[0] = [1, 3, 5, 8, 9, 10, 11, 12, 13]
    # SUCOrder[1] = [2]
    # SUCOrder[2] = [14]
    # SUCOrder[3] = [4]
    # SUCOrder[4] = [14]
    # SUCOrder[5] = [6]
    # SUCOrder[6] = [7]
    # SUCOrder[7] = [14]
    # SUCOrder[8] = [14]
    # SUCOrder[9] = [14]
    # SUCOrder[10] = [17]
    # SUCOrder[11] = [14]
    # SUCOrder[12] = [17]
    # SUCOrder[13] = [17]
    # SUCOrder[14] = [15, 16]
    # SUCOrder[15] = [17]
    # SUCOrder[16] = [17]
    # SUCOrder[17] = [18]
    # SUCOrder[18] = []

    # SUCOrder[0] = [1, 3, 5, 7, 10]
    # SUCOrder[1] = [2]
    # SUCOrder[2] = [9]
    # SUCOrder[3] = [4]
    # SUCOrder[4] = [9]
    # SUCOrder[5] = [6]
    # SUCOrder[6] = [11]
    # SUCOrder[7] = [8]
    # SUCOrder[8] = [9]
    # SUCOrder[9] = [11]
    # SUCOrder[10] = [15]
    # SUCOrder[11] = [12, 13]
    # SUCOrder[12] = [14]
    # SUCOrder[13] = [14]
    # SUCOrder[14] = [15]
    # SUCOrder[15] = [16]
    # SUCOrder[16] = []

    # #工序顺序
    PREOrder = defaultdict(lambda: [])
    PREOrder[0] = []
    PREOrder[1] = [0]
    PREOrder[2] = [0]
    PREOrder[3] = [1,2]
    PREOrder[4] = [3]
    PREOrder[5] = [3]
    PREOrder[6] = [3]
    PREOrder[7] = [3]
    PREOrder[8] = [5]
    PREOrder[9] = [5,6]
    PREOrder[10] = [7]
    PREOrder[11] = [8]
    PREOrder[12] = [4,11,9,10]
    PREOrder[13] = [12]
    # PREOrder[0] = []
    # PREOrder[1] = [0]
    # PREOrder[2] = [1]
    # PREOrder[3] = [0]
    # PREOrder[4] = [3]
    # PREOrder[5] = [0]
    # PREOrder[6] = [5]
    # PREOrder[7] = [6]
    # PREOrder[8] = [0]
    # PREOrder[9] = [0]
    # PREOrder[10] = [0]
    # PREOrder[11] = [0]
    # PREOrder[12] = [0]
    # PREOrder[13] = [0]
    # PREOrder[14] = [2,4,7,8,9,11]
    # PREOrder[15] = [14]
    # PREOrder[16] = [14]
    # PREOrder[17] = [15,16]
    # PREOrder[18] = [17]
    # OrderInputMes = [
    #     [(0, 1), (0, 0), (0, 0)],  # 2
    #     [(0, 1), (0, 0), (0, 1)],  # 3
    #     [(1, 1), (0, 0), (0, 0)],  # 4
    #     [(1, 2), (0, 0), (0, 1)],  # 5
    #     [(0, 1), (0, 0), (0, 0)],  # 6
    #     [(3, 2), (0, 1), (0, 1)],  # 7
    #     [(2, 2), (0, 0), (0, 0)],  # 8,
    #     [(2, 1), (0, 1), (0, 1)],  # 9
    #     [(3, 1), (4, 1), (0, 0)],  # 10
    #     [(3, 1), (3, 1), (0, 0)],  # 11
    #     [(3, 2), (0, 0), (0, 1)],  # 12
    #     [(3, 1), (0, 0), (0, 0)],  # 13
    #     [(3, 1), (0, 0), (0, 0)],  # 14
    #     [(0, 1), (2, 1), (0, 0)],  # 15
    #     [(2, 1), (0, 0), (0, 0)],  # 16
    # ]
    # 特设、航电、军械、机械

    OrderInputMes = [
        [(1, 1), (0, 0), (0, 0)],  # 1 通风
        [(0, 1), (0, 0), (0, 1)],  # 2 开电
        [(1, 1), (1, 1), (0, 0)],  # 3 机翼展开
        [(2, 2), (0, 0), (0, 1)],  # 4 数据加载
        [(3, 1), (2, 1), (0, 0)],  # 5 充氧
        [(3, 1), (3, 1), (0, 0)], # 6充氮
        [(0, 2), (0, 0), (0, 1)],  # 7 外观检查
        [(2, 2), (0, 0), (0, 0)],  # 8 挂弹
        [(3, 2), (0, 1), (0, 0)],  # 9 加油
        [(0, 2), (0, 0), (0, 0)],  # 10 装填航炮
        [(2, 1), (0, 0), (0, 0)],  # 11 弹药加载
        [(1, 1), (1, 1), (0, 0)],  # 12 机翼折叠
    ]

    # OrderInputMes = [
    #     [(0, 1), (0, 0), (0, 0)],  # 2
    #     [(0, 1), (1, 1), (0, 1)],  # 3
    #     [(1, 1), (0, 0), (0, 0)],  # 4
    #     [(1, 1), (1, 1), (0, 1)],  # 5
    #     [(2, 1), (0, 0), (0, 0)],  # 6
    #     [(2, 2), (1, 1), (0, 1)],  # 7
    #     [(2, 2), (0, 0), (0, 0)],  # 8,
    #     [(3, 1), (0, 1), (0, 1)],  # 9
    #     [(3, 1), (4, 1), (0, 0)],  # 10
    #     [(3, 1), (3, 1), (0, 0)],  # 11
    #     [(3, 2), (1, 1), (0, 1)],  # 12
    #     [(3, 1), (0, 0), (0, 0)],  # 13
    #     [(3, 1), (0, 0), (0, 0)],  # 14
    #     [(0, 1), (2, 1), (0, 0)],  # 15
    #     [(2, 1), (0, 0), (0, 0)],  # 16
    #     [(2, 1), (0, 0), (0, 0)],  # 17
    #     [(1, 2), (0, 0), (0, 0)],  # 18
    # ]


    VACP = [0,
            0,  # 虚拟1
            2,  # 2 特设外观检查#供电
            1.5,  # 3 特设座舱检查
            1.5,  # 4 航电外观检查
            1.5,  # 5 航电座舱检查
            1.5,  # 6 军械外观检查
            1.5,  # 7 军械座舱检查
            2,  # 8 航空弹药加载
            2.5,  # 9 添加燃油
            2,  # 10 添加液压油
            2.5,  # 11 充氮
            1.5,  # 12 机械座舱检查
            1.5,  # 13 机械外观检查
            2,  # 14 发动机检查
            2.5,  # 15 充氧
            2.5,  # 16 挂弹
            2.5,  # 17 挂弹
            1.5,  # 18 惯导
            0  # 19
            ]
    # OrderTimeMax = [5,
    #                 7,
    #                 6,
    #                 9,
    #                 6,
    #                 8,
    #                 17,
    #                 8,
    #                 6,
    #                 6,
    #                 16,
    #                 10,
    #                 6,
    #                 12,
    #                 12,
    #                 8
    #                 ]
    # OrderTimeMax = [5,
    #                 7,
    #                 6,
    #                 9,
    #                 6,
    #                 8,
    #                 17,
    #                 8,
    #                 6,
    #                 6,
    #                 16,
    #                 10,
    #                 6,
    #                 12,
    #                 12,
    #                 8
    #                 ]

    lowTime = 90  # 不能超过90 min

def getTime(i):
        # 定义每种任务的时间分布
            if i == 0:
                return 0
            elif i == 1:
                return 0
            elif i==2:
                return random.randint(1, 6)
            elif i == 3:
                return random.randint(3, 8)
            elif i == 4:
                return random.randint(1, 6)
            elif i == 5:
                return random.randint(4, 9)
            elif i == 6:
                return random.randint(3, 7)
            elif i == 7:
                return random.randint(3, 8)
            elif i == 8:
                return random.randint(3, 8)
            elif i == 9:
                return random.randint(10, 17)
            elif i == 10:
                return random.randint(3, 8)
            elif i == 11:
                return random.randint(2, 6)
            elif i == 12:
                return random.randint(2, 6)
            elif i == 13:
                return random.randint(10, 16)
            elif i == 14:
                return random.randint(6, 10)
            elif i == 15:
                return random.randint(1, 6)
            elif i == 16:
                return random.randint(6, 10)
            elif i == 17:
                return random.randint(6, 10)
            elif i == 18:
                return random.randint(2, 7)
            elif i == 19:
                return 0










