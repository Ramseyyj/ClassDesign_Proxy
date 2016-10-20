import numpy as np

file_score = open('douban.txt', 'r')
file_relation = open('follow(in-out).txt', 'r')

users = set([])
movies = set([])

for line in file_score:
    line = line.strip('\n').split('::')
    users.add(line[0])
    movies.add(line[1])

print(len(users))
print(len(movies))

m = np.array([[0, 0],
              [0, 0]])

for line in file_relation:
    line = line.strip('\n').split()
    if line[0] in users:
        if line[1] in users:
            m[0, 0] += 1
        else:
            m[0, 1] += 1
    else:
        if line[1] in users:
            m[1, 0] += 1
        else:
            m[1, 1] += 1

print(m)

file_score.close()
file_relation.close()
