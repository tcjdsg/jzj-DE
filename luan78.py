import numpy as np

from DE_archive_8 import JADE
from initm import InitM

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

    zuhe = [[1,1,1,1]]

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

                model.Repetition()

