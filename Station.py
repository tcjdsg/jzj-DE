import copy


class Station:
    def __init__(self,info):
        self.type = info[0]
        self.zunumber = info[1]
        self.number = info[2]
        self.state = 0
        self.NowJZJ = 0
        self.NowTaskId = 0

        self.alreadyworkTime = 0
        self.OrderOver = []  # 已完成工序
        #已完成工序
        self.TaskWait = [] #待完成工序
        self.fugai_time=0

    def delete(self,Activity):
        self.alreadyworkTime -= Activity.duration
        self.OrderOver.remove(Activity)
        self.OrderOver.sort(key=lambda x: x.es)
    def update(self,Activity):
        self.state=1
        self.alreadyworkTime += Activity.duration
        self.OrderOver.append(Activity)
        self.OrderOver.sort(key=lambda x: x.es)



