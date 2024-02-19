import copy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from FixedMess import FixedMes
plt.rcParams['font.family'] = 'SimHei'



def draw_h_s(humans,stations,jzjNums):
    font = {
        'weight': 'normal',
        'size': 17,
    }
    assert jzjNums==8 or jzjNums==12
    if jzjNums==8:
        name_human = ['$Lp_{1}^{1}$  ', '$Lp_{1}^{2}$  ', '$Lp_{1}^{3}$  ', '$Lp_{1}^{4}$  ', '$Lp_{1}^{5}$  ', '$Lp_{1}^{6}$  ',
                      '$Lp_{2}^{1}$  ', '$Lp_{2}^{2}$  ', '$Lp_{2}^{3}$  ', '$Lp_{2}^{4}$  ',
                      '$Lp_{3}^{1}$  ', '$Lp_{3}^{2}$  ', '$Lp_{3}^{3}$  ', '$Lp_{3}^{4}$  ', '$Lp_{3}^{5}$  ', '$Lp_{3}^{6}$  ',
                      '$Lp_{3}^{7}$  ', '$Lp_{3}^{8}$  ', '$Lp_{3}^{9}$  ', '$Lp_{3}^{10}$',
                      '$Lp_{4}^{1}$  ', '$Lp_{4}^{2}$  ', '$Lp_{4}^{3}$  ', '$Lp_{4}^{4}$  ', '$Lp_{4}^{5}$  '

                      ]
        jzj_numbers = [1,3,4,6,7,8,9,11]
    if jzjNums==12:
        name_human = ['$Lp_{1}^{1}$  ', '$Lp_{1}^{2}$  ', '$Lp_{1}^{3}$  ', '$Lp_{1}^{4 }$  ', '$Lp_{1}^{5 }$  ', '$Lp_{1}^{6 }$  ',
                      '$Lp_{1}^{7 }$  ', '$Lp_{1}^{8 }$  ',
                      '$Lp_{2}^{1 }$  ', '$Lp_{2}^{2 }$  ', '$Lp_{2}^{3 }$  ', '$Lp_{2}^{4 }$  ', '$Lp_{2}^{5 }$  ', '$Lp_{2}^{6 }$  ',
                      '$Lp_{3}^{1 }$  ', '$Lp_{3}^{2 }$  ', '$Lp_{3}^{3 }$  ', '$Lp_{3}^{4 }$  ', '$Lp_{3}^{5 }$  ', '$Lp_{3}^{6 }$  ',
                      '$Lp_{3}^{7 }$  ', '$Lp_{3}^{8 }$  ', '$Lp_{3}^{9}$  ', '$Lp_{3}^{10}$', '$Lp_{3}^{11}$',
                      '$Lp_{4}^{1 }$  ', '$Lp_{4}^{2 }$  ', '$Lp_{4}^{3 }$  ', '$Lp_{4}^{4 }$  ', '$Lp_{4}^{5 }$  ', '$Lp_{4}^{6 }$  ',
                      '$Lp_{4}^{7 }$  '
                      ]
        jzj_numbers = [1, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16]
    name_station = ['$Lr_{1}^{1 }$  ', '$Lr_{1}^{2 }$  ', '$Lr_{1}^{3 }$  ', '$Lr_{1}^{4 }$  ', '$Lr_{1}^{5 }$  ', '$Lr_{1}^{6 }$  ',
                    '$Lr_{1}^{7 }$  ',
                    '$Lr_{2}^{1 }$  ', '$Lr_{2}^{2 }$  ', '$Lr_{2}^{3 }$  ', '$Lr_{2}^{4 }$  ', '$Lr_{2}^{5 }$  ', '$Lr_{2}^{6 }$  ',
                    '$Lr_{2}^{7 }$  ', '$Lr_{2}^{8 }$  ', '$Lr_{2}^{9 }$  ', '$Lr_{2}^{10}$', '$Lr_{2}^{11}$',
                    '$Lr_{2}^{12}$',
                    '$Lr_{2}^{13}$', '$Lr_{2}^{14}$', '$Lr_{2}^{15}$', '$Lr_{2}^{16}$',
                    '$Lr_{3}^{1 }$  ', '$Lr_{3}^{2 }$  ', '$Lr_{3}^{3 }$  ', '$Lr_{3}^{4 }$  ', '$Lr_{3}^{5 }$  ', '$Lr_{3}^{6 }$  ',
                    '$Lr_{4}^{1 }$  ', '$Lr_{4}^{2 }$  ', '$Lr_{4}^{3 }$  ', '$Lr_{4}^{4 }$  ', '$Lr_{4}^{5 }$  ', '$Lr_{4}^{6 }$  ',
                    '$Lr_{5}^{1 }$ ', '$Lr_{5}^{2 }$ ', '$Lr_{5}^{3 }$  ', '$Lr_{5}^{4 }$  ', '$Lr_{5}^{5 }$  ', '$Lr_{5}^{6 }$  ',
                    '$Lr_{5}^{7 }$ ', '$Lr_{5}^{8 }$ '
                    ]

    Draw1(humans, name_human,font,jzj_numbers)
    Draw2(stations, name_station,font,jzj_numbers)
def Draw1(all_people,name,font,jzj_numbers):
    plt.figure(figsize=(8, 13))
    colors = ['lightcoral', 'brown', 'red', 'sandybrown', 'teal', 'green', 'limegreen', 'olive','steelblue','royalblue','c','deepskyblue','thistle','violet','purple','deeppink']

    c = []
    number=-1
    for i in range(len(all_people)):
        for j in range(len(all_people[i])):
            number += 1
            for order in all_people[i][j].OrderOver:
                job = order.belong_plane_id

                gongxu = (order.taskid -1) % FixedMes.planeOrderNum +1
                id = order.id
                time1= order.es
                time2= order.ef
                c.append([jzj_numbers.index(job) + 1, job, gongxu, time2 - time1])
                if (time2 - time1) != 0:
                   plt.barh(number, time2 - time1-0.05,
                     left=time1, color=colors[jzj_numbers.index(job)],height=0.8)
                news =  str(gongxu)
                infmt = news
                if (time2 - time1)!=0 and (time2-time1)>=2:
                   plt.text(x=time1, y=number-0.3, s=infmt, fontdict={ 'size' : 14},
                       color='white',)
                if  (time2 - time1)!=0 and (time2-time1)<2:
                    plt.text(x=time1, y=number - 0.3, s=infmt, fontdict={ 'size' : 14},
                             color='white')

    patches = [mpatches.Patch(color=colors[jzj_numbers.index(i)],linewidth=2,label='$I$' +'-'+ str(jzj_numbers.index(i)+1)) for i in jzj_numbers]

    plt.yticks([i for i in range(number+1)], name, font = {'size': 11})
    if len(jzj_numbers) == 8:

        plt.xticks([i*5 for i in range(13)], [i*5 for i in range(13)], font = font)
    if len(jzj_numbers) == 12:
        plt.xticks([i * 5 for i in range(16)], [i * 5 for i in range(16)], font = font)


    plt.xlabel("t/min",font)
    plt.legend(handles=patches)
    plt.show()
    # plt.yticks([i + 1 for i in range(people_number)])
def Draw2(all_people,name,font,jzj_numbers):
    plt.figure(figsize=(8, 13))
    colors = ['lightcoral', 'brown', 'red', 'sandybrown', 'teal', 'green', 'limegreen', 'olive', 'steelblue',
              'royalblue', 'c', 'deepskyblue', 'thistle', 'violet', 'purple', 'deeppink']

    num = -1
    tmp_name = copy.deepcopy(name)
    for i in range(len(all_people)):
        for j in range(len(all_people[i])):
            num +=1
            if len(all_people[i][j].OrderOver)==0:
                tmp_name.remove(name[num])
    name = copy.deepcopy(tmp_name)

    jzj_numbers = sorted(jzj_numbers, key=lambda x: x)
    print(jzj_numbers)
    c = []
    number = -1
    for i in range(len(all_people)):
        for j in range(len(all_people[i])):
            if len(all_people[i][j].OrderOver)==0:
                continue
            number += 1

            for order in all_people[i][j].OrderOver:
                job = order.belong_plane_id
                #         print(job)
                gongxu = (order.taskid - 1) % FixedMes.planeOrderNum + 1
                id = order.id
                time1 = order.es
                time2 = order.ef
                c.append([jzj_numbers.index(job) + 1, job, gongxu, time2 - time1])
                if (time2 - time1) != 0:
                    plt.barh(number, time2 - time1 - 0.05,
                             left=time1, color=colors[jzj_numbers.index(job)])

                    news = str(gongxu)
                    infmt = news
                    if (time2 - time1) != 0 and (time2 - time1) >= 2:
                        plt.text(x=time1, y=number - 0.3, s=infmt,
                                 fontdict={ 'size': 14},
                                 color='white', )
                    if (time2 - time1) != 0 and (time2 - time1) < 2:
                        plt.text(x=time1, y=number - 0.3, s=infmt,
                                 fontdict={ 'size': 14},
                                 color='white')

    patches = [mpatches.Patch(color=colors[jzj_numbers.index(i)], linewidth=2,
                                  label='$I$' + '-' + str(jzj_numbers.index(i) + 1)) for i in jzj_numbers]

    plt.yticks([i for i in range(number+1)], name, font={'size': 11})
    if len(jzj_numbers) == 8:
            plt.xticks([i*5 for i in range(13)], [i*5 for i in range(13)], font=font)
    if len(jzj_numbers) == 12:
            plt.xticks([i * 5 for i in range(16)], [i * 5 for i in range(16)], font=font)


    plt.xlabel("t/min",font)
    plt.legend(handles=patches)
    plt.show()




