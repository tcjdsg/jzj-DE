import numpy as np


class Chromosome(object):
    def __init__(self,codes,f):
        self.codes = np.array(codes)
        self.fitness=f    #适应度
        self.rank = -1    #用于多目标
        self.crowding_distance = -1


    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def setcodes(self,codes):
            self.codes=codes


    def __gt__(self, other):
        if self.rank > other.rank:
            return True
        if self.rank==other.rank and self.crowding_distance < other.crowding_distance:
            return True
        return  False
