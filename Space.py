import copy


class Space(object):
    def __init__(self,jzjId):
        self.type = jzjId

        self.state = 0
        self.NowJZJ = jzjId
        self.NowTaskId = 0

        self.alreadyworkTime = 0
        self.OrderOver = []  # 已完成工序
        #已完成工序
        self.TaskWait = [] #待完成工序


    def update(self,Activity):
        # self.alreadyworkTime += Activity.duration
        self.state = 1
        self.alreadyworkTime += Activity.duration
        self.OrderOver.append(copy.deepcopy(Activity))
        self.OrderOver.sort(key=lambda x: x.es)