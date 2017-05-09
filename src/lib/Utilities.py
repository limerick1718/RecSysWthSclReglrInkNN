import numpy
from lib import util

def rmse(validation_index, R, U, V):

    rmse = 0
    T = 0
    for index in xrange(len(validation_index)):
        sI,sJ =  validation_index[index].split(',')

        i = int(float(sI))
        j = int(float(sI))

        T += 1

        UT = U[i].T
        VT = V[j]
        RT = R[i][j]
        e = numpy.dot(UT, VT) - RT
        rmse += e ** 2

    return  numpy.sqrt(rmse/T)


def mae(validation_index, R, U, V):

    rmse = 0
    T = 0
    for index in xrange(len(validation_index)):
        sI,sJ =  validation_index[index].split(',')

        i = int(float(sI))
        j = int(float(sI))

        T += 1

        UT = U[i].T
        VT = V[j]
        RT = R[i][j]
        e = numpy.dot(UT, VT) - RT
        rmse += abs(e)

    return  rmse/T

def sr_f(i, P, SG):
    reg = 0

    for f in xrange(len(SG[i])):
        reg += SG[i][f] * (P[i] - P[f])

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

def load_grafo_social_for_anotherDataSet(R, social_network, userNumber):
    grafo_size = userNumber

    social_graph = numpy.zeros((grafo_size, grafo_size))
    social_graph_preSim = numpy.zeros((grafo_size, grafo_size))

    for i in xrange(len(social_network)):
        user = int(float(social_network[i][0]))
        friend = int(float(social_network[i][1]))

        if user < grafo_size:
            if friend < grafo_size:

                social_graph[user][friend] = 1
                social_graph[friend][user] = 1

                x = R[user]
                y = R[friend]

                cor_pearson = util.pearson(x, y)

                social_graph_preSim[user][friend] = cor_pearson
                social_graph_preSim[friend][user] = cor_pearson

    return social_graph, social_graph_preSim
