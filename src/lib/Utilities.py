import numpy
from lib import util

def rmse(validation_index, R, U, V):

    rmse = 0
    T = 0
    for index in xrange(len(validation_index)):
        now =  int(float(validation_index[index]))

        i = int(float(R[now][0]))
        j = int(float(R[now][1]))
        ratingScores = int(float(R[now][2]))

        T += 1

        UT = U[i].T
        VT = V[j]

        e = numpy.dot(UT, VT) - ratingScores
        rmse += e ** 2

    return  numpy.sqrt(rmse/T)


def mae(validation_index, R, U, V):

    rmse = 0
    T = 0
    for index in xrange(len(validation_index)):
        now =  int(float(validation_index[index]))
        i = int(float(R[now][0]))
        j = int(float(R[now][1]))
        ratingScores = int(float(R[now][2]))

        T += 1

        UT = U[i].T
        VT = V[j]

        e = numpy.dot(UT, VT) - ratingScores
        rmse += abs(e)

    return  rmse/T

def sr_f(i, P, SG):
    reg = 0

    for j in len(SG[i]):
        reg += SG[i][j][1] * (P[i] - P[SG[i][j][0]])
        # reg += SG[i][f] * (P[i] - P[f])

    return reg


def sr_fkNN(i, P, SG, weightVector, imitates):
    reg = 0

    for f in imitates:
        reg += SG[i][f] * (P[i] - P[f]) * weightVector[f]

    return reg

def load_matrix_index(R, bound):
    list_index = []
    validation_index = []
    for i in xrange(len(R)):
        for j in xrange(len(R[0])):
            if R[i][j] > 0:
                randNumber=numpy.random.randint(0,100)
                if randNumber <= bound:
                    list_index.append(`i`+','+`j`)
                else:
                    validation_index.append(`i`+','+`j`)
    return list_index, validation_index

def load_matrix_index_for_anotherDataSet(R, bound, userNumber, itemNumber):
    list_index = []
    validation_index = []
    newR = numpy.zeros((userNumber, itemNumber))
    for i in xrange(len(R)):
        x = R[i][0]
        y = R[i][1]
        value = R[i][2]/1000
        if x<userNumber:
            newR[x][y] = value
            randNumber = numpy.random.randint(0, 100)
            if randNumber <= bound:
                list_index.append(`x` + ',' + `y`)
            else:
                validation_index.append(`x` + ',' + `y`)

    return list_index, validation_index, newR

def load_matrix_index_for_anotherDataSet_new(R, bound, userNumber):
    list_index = []
    validation_index = []
    newR = [[] for j in range(userNumber + 1)]
    for i in xrange(len(R)):
        x = int(float(R[i][0]))
        R[i][2] = R[i][2] * 1.0 /10000
        # Change to proper scale of number
        newR[x].append([R[i][1], R[i][2]/10000])
        randNumber = numpy.random.randint(0, 100)
        if randNumber <= bound:
            list_index.append(`i`)
        else:
            validation_index.append(`i`)

    return list_index, validation_index, newR

def load_grafo_social(R, social_network):
    grafo_size = len(R)

    social_graph = numpy.zeros((grafo_size, grafo_size))


    for i in xrange(len(social_network)):
        user = int(float(social_network[i][0]))
        friend = int(float(social_network[i][1]))

        x = R[user]
        y = R[friend]

        cor_pearson = util.pearson(x, y)

        social_graph[user][friend] = cor_pearson
        social_graph[friend][user] = cor_pearson

    # print social_graph

    return social_graph


def load_grafo_social_for_anotherDataSet_old(R, SN_FILE):
    grafo_size = 2100

    social_graph = numpy.zeros((grafo_size, grafo_size))

    social_network = numpy.loadtxt(open(SN_FILE, "rb"), delimiter='\t')

    for i in xrange(len(social_network)):
        user = float(social_network[i][0])
        friend = float(social_network[i][1])

        if user < grafo_size:
            if friend < grafo_size:
                x = R[user]
                y = R[friend]

                cor_pearson = util.pearson(x, y)

                social_graph[user][friend] = cor_pearson
                social_graph[friend][user] = cor_pearson

    # print social_graph

    return social_graph

def load_grafo_social_for_anotherDataSet(newR, social_network, userNumber, itemNumber):
    grafo_size = userNumber

    social_graph = [[] for i in xrange(userNumber + 1)]
    social_graph_preSim = [[] for i in xrange(userNumber + 1)]
    userVector = numpy.zeros((itemNumber + 1, 1))
    friendVector = numpy.zeros((itemNumber + 1, 1))

    for i in xrange(len(social_network)):
        user = int(float(social_network[i][0]))
        friend = int(float(social_network[i][1]))

        userItemsRating = newR[user]
        for j in xrange(len(userItemsRating)):
            userVector[int(float(userItemsRating[j][0]))] = float(userItemsRating[j][1])
        friendsItemsRating = newR[friend]
        for k in xrange(len(friendsItemsRating)):
            friendVector[int(float(friendsItemsRating[k][0]))] = float(friendsItemsRating[k][1])

        if user <= grafo_size:
            if friend > user:

                social_graph[user].append(friend)
                social_graph[friend].append(user)

                cor_pearson = util.pearson(userVector, friendVector)

                social_graph_preSim[user].append([friend,cor_pearson])
                social_graph_preSim[friend].append([user,cor_pearson])
    print '*******************load social graph success************************'
    return social_graph, social_graph_preSim
