import numpy as np
import math


def users_topics_count(filename):

    file_score = open(filename, 'r')

    users_set = set([])
    topics_set = set([])

    for line in file_score:
        line = line.strip('\n').split('::')
        users_set.add(line[0])
        topics_set.add(line[1])

    users_count = len(users_set)
    topics_count = len(topics_set)

    file_score.close()

    return users_count, topics_count


def interested_matrix_compute(users_count, topics_count):

    file_score = open('douban1.txt', 'r')

    users = []
    topics = []
    m = np.zeros((users_count, topics_count))

    for line in file_score:
        line = line.strip('\n').split('::')

        if line[0] not in set(users):
            users.append(line[0])

        if line[1] not in set(topics):
            topics.append(line[1])

        user_index = users.index(line[0])
        topic_index = topics.index(line[1])

        m[user_index, topic_index] = line[2]

    file_score.close()

    return m


def density_compute(m):

    users_des = []
    for i in range(m.shape[0]):
        temp_des = 0
        for j in range(m.shape[1]):
            if m[i, j] != 0.0:
                temp_des += 1
        users_des.append(temp_des / m.shape[1])

    return users_des


# 选取核心用户
def core_users_select(matrix):

    density = density_compute(matrix)

    core_users_list = []
    for i in range(len(density)):
        if density[i] > const_lambda:
            core_users_list.append(i)

    return core_users_list


def Psj_compute(matrix, Ci_set):

    Psj = np.zeros((matrix.shape[1]))

    for user in Ci_set:
        for sj in range(matrix.shape[1]):
            if matrix[user, sj] > 0.0:
                Psj[sj] += 1

    return Psj


def pij_compute(matrix, Ci_set, Psj):

    Ci_user_count = len(Ci_set)

    pij = np.zeros((matrix.shape[1]))

    for user in Ci_set:
        for sj in range(matrix.shape[1]):
            if matrix[user, sj] > 0.0:
                Psj[sj] += 1

    for i in range(matrix.shape[1]):
        pij[i] = Psj[i] / Ci_user_count

    return pij

# 计算Ci_set聚类的模糊度Ambi
def Ambi_compute(matrix, Ci_set, Ambij_flag):

    Ci_user_count = len(Ci_set)

    Ambi = 0
    Psj = Psj_compute(matrix, Ci_set)
    pij = pij_compute(matrix, Ci_set, Psj)
    Ambij = np.zeros((matrix.shape[1]))

    for i in range(matrix.shape[1]):

        if pij[i] >= const_delta:
            Ambij[i] = Ci_user_count - Psj[i]
        else:
            Ambij[i] = Psj[i]

        Ambi += Ambij[i]

    if Ambij_flag:
        return Ambij, Ambi
    else:
        return Ambi


# 计算全局模糊度Amb
def Amb_compute(matrix, Clus):

    Ambi_sum = 0

    for Ci_set in Clus:

        Ambi = Ambi_compute(matrix, Ci_set, False)
        Ambi_sum += Ambi

    return math.log(Ambi_sum / len(Clus))


# 计算Ci_set聚类在每个主题上的兴趣度caij
def caij_compute(matrix, Ci_set):

    caij = np.zeros((matrix.shape[1]))
    Psj = Psj_compute(matrix, Ci_set)
    pij = pij_compute(matrix, Ci_set, Psj)

    for i in range(matrix.shape[1]):

        if pij[i] >= const_delta:

            temp_sum = 0
            for user in Ci_set:
                temp_sum += matrix[user, i]

            caij = temp_sum / len(Ci_set)

        else:
            caij = 0

    return caij


def diff_compute(matrix, Ci_set, Cj_set):

    cvi = caij_compute(matrix, Ci_set)
    cvj = caij_compute(matrix, Cj_set)

    diff = cvi.dot(cvj) / (np.linalg.norm(cvi) * np.linalg.norm(cvj))

    return diff


def dvst_compute(matrix, Clus):

    temp_sum = 0
    for Ci in Clus:
        for Cj in Clus:
            temp_sum += diff_compute(matrix, Ci, Cj)

    dvst = temp_sum / len(Clus)

    return dvst


def Cluster_split(matrix, Ci, target):

    Ci1 = set([])
    Ci2 = set([])

    for user in Ci:
        if matrix[user, target] > 0:
            Ci1.add(user)
        else:
            Ci2.add(user)

    return Ci1, Ci2


# 图摘要算法
def SNAP_cluster_algorithm(matrix, k_iteration):

    core_users = core_users_select(matrix)
    Clus = [set(core_users)]

    maxAmb = 0
    srcCi = 0
    target = 0
    Amb = 1
    k = 0

    while Amb == 0 or k == k_iteration:
        k += 1
        for i in range(len(Clus)):

            Ambij, Ambi = Ambi_compute(matrix, Clus[i], True)
            sj = np.argmax(Ambij)
            if Ambi > maxAmb:
                maxAmb = Ambi
                target = sj
                srcCi = i

        Ci1, Ci2 = Cluster_split(matrix, Clus[srcCi], target)
        Clus.pop(srcCi)
        Clus.append(Ci1)
        Clus.append(Ci2)

        Amb = Amb_compute(matrix, Clus)
        dvst = dvst_compute(matrix, Clus)

    return Clus, Amb, dvst


usersCount, topicCount = users_topics_count('douban1.txt')
interestedMatrix = interested_matrix_compute(usersCount, topicCount)


# 强度阈值
const_delta = 0.001

# 密度阈值
const_lambda = 0.001


