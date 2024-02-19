from matplotlib import pyplot as plt
import numpy as np
import math


def sigmoid_function(z):
    fz = []
    for num in z:
        fz.append(5 / (1 + math.exp(-num)))
    return fz


if __name__ == '__main__':
    # best_8=np.load("result/zhengjiao_best__12_12_instance4.npy")
    # mean_8 = np.load("result/zhengjiao_mean__12_12_instance4.npy")
    # print()
    # # i = [1,1,2,1]
    # # print(i.index(min(i)))
    instance = np.load(f'dataset/DE_12_12_instance4.npy', allow_pickle=True)[0]
    fea = instance[1]
    list_fea= []
    for i in range(fea.shape[0]-1):
        list_fea.append(fea[i])
    list_fea = sorted(list_fea,key=lambda x:x[6])

    adj = np.load(f'dataset/problems_9_12.npy', allow_pickle=True)[0][0]
    fea_8 = list_fea[:8*12+1]
    jzjnew = list_fea[1:13]
    for feaorder in jzjnew:
        feaorder [6] = 5
    fea_8 = np.concatenate((fea_8,list_fea[1:13]),axis=0)
    fea_8 = np.concatenate((fea_8,fea[-1:]),axis=0)
    data = []
    human_r = [6,4,10,5]
    data.append((adj, fea_8, human_r))

    np.save(f'dataset/problems_add_jzj.npy', np.asarray(data, dtype=object))
    #
    # instance = np.load(f'dataset/problems_12_12.npy', allow_pickle=True)[4]
    # fea = instance[1]
    # adj = np.load(f'dataset/problems_12_12.npy', allow_pickle=True)[4][0]
    #
    # data = []
    # human_r = [8,6,11,7]
    # data.append((adj, fea, human_r))
    #
    #
    # np.random.shuffle(data)
    #
    # np.save(f'dataset/DE_12_12_instance4.npy', np.asarray(data, dtype=object))
    #
    #
