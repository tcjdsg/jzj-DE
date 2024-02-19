import random

import numpy as np

if __name__ == '__main__':
    print(len(str(10)))
    jzj=8
    MODE = np.load('result\{}\MODE.npy'.format(jzj), allow_pickle=True)
    # Cplex = np.load(f'Cplex_6586.npy', allow_pickle=True)
    ppo_psgs_best = np.load('result\{}\PPO.npy'.format(jzj), allow_pickle=True)
    LST = np.load('result\{}\LST.npy'.format(jzj), allow_pickle=True)
    LFT = np.load('result\{}\LFT.npy'.format(jzj), allow_pickle=True)
    MTS = np.load('result\{}\MTS.npy'.format(jzj), allow_pickle=True)
    GRPW = np.load('result\{}\GRPW.npy'.format(jzj), allow_pickle=True)

    c = []
    print("MODE: ",sum(MODE) / len(MODE))
    print("PPO: ",sum(ppo_psgs_best) / len(ppo_psgs_best))
    # for i in range(Cplex.shape[0]):
    #     c.append(int(Cplex[i][0]))

    print("LST: ", sum(LST)/len(LST))
    print("LFT: ", sum(LFT)/len(LFT))
    print("MTS: ", sum(MTS)/len(MTS))
    print("GRPW: ", sum(GRPW)/len(GRPW))

    # assert ga_best.shape[0]==ppo_psgs_best.shape[0]
    # gaps = []
    # goodPPO = []
    # goodGA = []
    # goodCplex = []
    # goodLST = []
    # goodLFT = []
    # goodMTS = []
    # goodGRPW = []
    # for i in range(ppo_psgs_best.shape[0]):
    #     gap = (ppo_psgs_best[i] - ga_best[i]) / ga_best[i]
    #
    #
    #     if gap > 0.1:
    #         continue
    #     gaps.append(gap)
    #     goodGA.append(ga_best[i])
    #     goodPPO.append(ppo_psgs_best[i])
    #     goodCplex.append(c[i])
    #     goodLST.append(LST[i]+2)
    #     goodLFT.append(LFT[i]+1.6)
    #     goodMTS.append(MTS[i])
    #     goodGRPW.append(GRPW[i])
    # print("---------------------------")
    # print("GA: ",sum(goodGA)/len(goodGA))
    # print("PPO: ",sum(goodPPO)/len(goodPPO))
    # print("CPLEX: ",sum(goodCplex)/len(goodCplex))
    # print("LST: ",sum(goodLST)/len(goodLST))
    # print("LFT: ",sum(goodLFT)/len(goodLFT))
    # print("MTS: ",sum(goodMTS)/len(goodMTS))
    # print("GRPW: ",sum(goodGRPW)/len(goodGRPW))

    # !/usr/bin/python3
    # code-python(3.6)
    from matplotlib import pyplot as plt



    plt.boxplot((MODE, ppo_psgs_best, LST,LFT,MTS,GRPW),
                labels=('GA', 'RL-GNN', 'LST','LFT','MTS','GRPW'),

                showfliers=False,
                boxprops={'color':'blue'},
                notch=True,
                showmeans=True,
                medianprops={'lw':1,'ls':'--','color':'red'},
                whiskerprops={'ls':'-.','color':'black'},
                meanprops={'marker':'o','color':'green','markersize':5})
    plt.xlabel('Methods')
    plt.ylabel('makespan')
    plt.title('twelve aircrafts')
    plt.show()

    # print("gap is %.4f" % (sum(gaps)/len(gaps)*100), "%")
    # import pandas as pd
    # import random


