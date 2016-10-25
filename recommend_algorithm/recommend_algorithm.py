import numpy as np
import math
import random

const_maxfloat = 1.7976931348623157e+308

# 强度阈值
const_delta = 0.001

# 密度阈值
const_lambda = 0.008


# 生成一个row*column的矩阵，值范围为
def random_matrix(row, column):

    m = np.zeros((row, column))

    for i in range(row):
        for j in range(column):
            x = int(random.randint(0, 5))
            m[i, j] = x

    return m


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
            Ambij[i] = abs(Ci_user_count - Psj[i])
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

            caij[i] = temp_sum / len(Ci_set)

        else:
            caij[i] = 0

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
    dvst = 1
    k = 0

    while Amb == 0 or k != k_iteration:
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
        if len(Ci1) != 0:
            Clus.append(Ci1)
        if len(Ci2) != 0:
            Clus.append(Ci2)

        Amb = Amb_compute(matrix, Clus)
        dvst = dvst_compute(matrix, Clus)

    return Clus, Amb, dvst


# 全用户聚类
def All_users_cluster(matrix, core_cluster):

    all_users = set([i for i in range(matrix.shape[0])])
    core_users = core_users_select(matrix)
    non_core_users = all_users - set(core_users)

    cvi = np.zeros((len(core_cluster), matrix.shape[1]))
    for i in range(len(core_cluster)):
        caij = caij_compute(matrix, core_cluster[i])
        cvi[i] = caij

    for user in non_core_users:
        min_dist = const_maxfloat
        min_index = 0

        for j in range(len(cvi)):
            dist = np.linalg.norm(cvi[j] - matrix[user, :])

            if dist < min_dist:
                min_dist = dist
                min_index = j

        core_cluster[min_index].add(user)

    return core_cluster


def devij_compute(matrix, Gclus, si, sj):

    temp_sum = 0
    for i in range(len(Gclus)):
        cvi = caij_compute(matrix, Gclus[i])
        temp_sum += cvi[si] - cvi[sj]

    devij = temp_sum / len(Gclus)

    return devij


def recommand_matrix(matrix_train, Gclus):

    reMatrix = np.zeros((len(Gclus), matrix_train.shape[1]))
    preMatrix = np.zeros((len(Gclus), matrix_train.shape[1]))

    for i in range(len(Gclus)):

        cvi = caij_compute(matrix_train, Gclus[i])
        aver_cvi = np.mean(cvi)
        M = matrix_train.shape[1]
        cv1 = np.zeros(M)

        for j in range(M):
            if cvi[j] == 0.0:

                temp_sum = 0
                for k in range(M):
                    temp_sum += devij_compute(matrix_train, Gclus, j, k)

                cv1[j] = aver_cvi + temp_sum / (M - 1)
            else:
                cv1[j] = cvi[j]

        sorted_index = np.argsort(-cv1)
        reMatrix[i] = sorted_index
        preMatrix[i] = cv1

    return reMatrix, preMatrix


def recommand(reMatrix, clusterMatrix, vec_user, Top_k):

    min_dist = const_maxfloat
    min_index = 0

    for i in range(len(clusterMatrix)):

        dist = np.linalg.norm(vec_user - clusterMatrix[i])

        if dist < min_dist:
            min_dist = dist
            min_index = i

    return reMatrix[min_index, :Top_k], min_index


def recall_compute(matrix_train, Gclus, matrix_test, Top_k):

    M = matrix_train.shape[1]

    recall = np.zeros(matrix_test.shape[0])

    reMatrix = recommand_matrix(matrix_train, Gclus)[0]

    clusterMatrix = np.zeros((len(Gclus), M))
    for i in range(len(Gclus)):
        cvi = caij_compute(matrix_train, Gclus[i])
        clusterMatrix[i] = cvi

    for i in range(matrix_test.shape[0]):

        # 命中次数
        hit_count = 0

        # 关注的主题数
        follow_count = 0

        for j in range(M):

            if matrix_test[i, j] != 0.0:
                follow_count += 1
                temp_vec = np.zeros(M)
                for k in range(M):
                    temp_vec[k] = matrix_test[i, j]
                temp_vec[j] = 0.0

                reList = recommand(reMatrix, clusterMatrix, temp_vec, Top_k)[0]

                if j in set(reList):
                    hit_count += 1

        recall[i] = hit_count / follow_count

    aver_recall = np.mean(recall)

    return aver_recall


def RMSE_compute(matrix_train, Gclus, matrix_test, Top_k):

    M = matrix_train.shape[1]

    RMSE = np.zeros(matrix_test.shape[0])

    reMatrix, preMatrix = recommand_matrix(matrix_train, Gclus)

    clusterMatrix = np.zeros((len(Gclus), M))
    for i in range(len(Gclus)):
        cvi = caij_compute(matrix_train, Gclus[i])
        clusterMatrix[i] = cvi

    for i in range(matrix_test.shape[0]):

        # 关注的主题数
        follow_count = 0

        temp_sum = 0

        for j in range(M):

            if matrix_test[i, j] != 0.0:
                follow_count += 1
                temp_vec = np.zeros(M)
                for k in range(M):
                    temp_vec[k] = matrix_test[i, j]
                temp_vec[j] = 0.0

                reList, reIndex = recommand(reMatrix, clusterMatrix, temp_vec, Top_k)

                if j in set(reList):
                    temp_sum += (preMatrix[reIndex, j] - matrix_test[i, j])**2
                else:
                    temp_sum += matrix_test[i, j]**2

        if follow_count != 0:
            RMSE[i] = math.sqrt(temp_sum / follow_count)
        else:
            RMSE[i] = 0

    aver_RMSE = np.mean(RMSE)

    return aver_RMSE


# usersCount, topicCount = users_topics_count('douban1.txt')
# interestedMatrix = interested_matrix_compute(usersCount, topicCount)
interestedMatrix = random_matrix(500, 50)
TOP_K = 30

N_g = interestedMatrix.shape[0]
Train_Matrix = interestedMatrix[:int(4 * N_g / 5), :]
Test_Matrix = interestedMatrix[:int(4 * N_g / 5) + 1, :]

core_cluster_g, Amb_g, dvst_g = SNAP_cluster_algorithm(Train_Matrix, 10)

all_users_cluster_g = All_users_cluster(Train_Matrix, core_cluster_g)

Recall_g = recall_compute(Train_Matrix, all_users_cluster_g, Test_Matrix, TOP_K)

RMSE_g = RMSE_compute(Train_Matrix, all_users_cluster_g, Test_Matrix, TOP_K)

print('recall: %r' % Recall_g)
print('RMSE: %r' % RMSE_g)

