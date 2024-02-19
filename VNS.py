# 基于模拟退火的插入领域
# def insert(self, opt, pop):
#     a = copy.deepcopy(pop)
#     self.inser(opt, a, FixedMes.act_info)
#     # MyInit.fitness(a, [], [], [])
#     return a
def inser(opt, pop, activities):

    preorder = activities[opt].predecessor
    success = activities[opt].successor

    ts = 0
    es = 999
    newcode = []
    newcode.append(pop.codes[0][:opt] + pop.codes[0][opt + 1:])
    newcode.append(pop.codes[1][:opt] + pop.codes[1][opt + 1:])

    # 得到了
    for id in preorder:
        if pop[0][id].es > ts:
            ts = activities[id].es

    for id in success:
        if activities[id].es < es:
            es = activities[id].es

    code = []

    for time in newcode[0]:
        if time[1] >= ts and time[1] <= es:
            code.append(time)

    qujian = sorted(code, key=lambda x: x[1])
    optnow = np.random.choice([x for x in range(0, len(qujian) - 1)], 1, replace=False)[0]
    time1 = qujian[optnow][1]
    time2 = qujian[optnow + 1][1]

    a = random.uniform(time1, time2)

    pop.codes[0][opt] = [opt, a]
    pop.codes[1][opt] = [opt, a + activities[opt].duration]
