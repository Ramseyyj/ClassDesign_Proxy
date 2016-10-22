import numpy as np

file_score = open('douban1.txt', 'r')
file_interested_matrix = open('interested_matrix.txt', 'w')
# file_relation = open('follow(in-out).txt', 'r')

users_set = set([])
movies_set = set([])

for line in file_score:
    line = line.strip('\n').split('::')
    users_set.add(line[0])
    movies_set.add(line[1])

users_count = len(users_set)
movies_count = len(movies_set)

file_score.close()

# m = np.array([[0, 0],
#               [0, 0]])
#
# for line in file_relation:
#     line = line.strip('\n').split()
#     if line[0] in users:
#         if line[1] in users:
#             m[0, 0] += 1
#         else:
#             m[0, 1] += 1
#     else:
#         if line[1] in users:
#             m[1, 0] += 1
#         else:
#             m[1, 1] += 1
#
# print(m)

file_score = open('douban1.txt', 'r')

users = []
topics = []
m = np.zeros((users_count, movies_count))

for line in file_score:
    line = line.strip('\n').split('::')

    if line[0] not in set(users):
        users.append(line[0])

    if line[1] not in set(topics):
        topics.append(line[1])

    user_index = users.index(line[0])
    topic_index = topics.index(line[1])

    m[user_index, topic_index] = line[2]

for i in range(m.shape[0]):
    des = 0
    for j in range(m.shape[1]):
        if m[i, j] != 0.0:
            des += 1
        print('%r ' % (m[i, j]), end='', file=file_interested_matrix)
    print('%f\n' % (des / m.shape[1]), file=file_interested_matrix)

print(m.shape)

file_score.close()
file_interested_matrix.close()
# file_relation.close()
