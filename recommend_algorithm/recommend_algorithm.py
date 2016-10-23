import numpy as np


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


# 计算Ci_set聚类的模糊度Ambi
def Ambi_compute(matrix, Ci_set):

    Ci_user_count = len(Ci_set)

    Ambi = 0
    Psj = np.zeros((matrix.shape[1]))
    pij = np.zeros((matrix.shape[1]))
    Ambij = np.zeros((matrix.shape[1]))

    for user in Ci_set:
        for sj in range(matrix.shape[1]):
            if matrix[user, sj] > 0.0:
                Psj[sj] += 1

    for i in range(matrix.shape[1]):
        pij[i] = Psj[i] / Ci_user_count

        if pij[i] >= const_delta:
            Ambij[i] = Ci_user_count - Psj[i]
        else:
            Ambij[i] = Psj[i]

        Ambi += Ambij[i]

    return Ambi


# 计算全局模糊度Amb
def Amb_compute(matrix, Clus):




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
        for Ci in Clus:





usersCount, topicCount = users_topics_count('douban1.txt')
interestedMatrix = interested_matrix_compute(usersCount, topicCount)


# 强度阈值
const_delta = 0.001

# 密度阈值
const_lambda = 0.001


