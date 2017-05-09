import numpy
from random import randint
from lib import mtxfac
from lib import util
from lib import stratus
from math import isnan


def load_grafo_social(R, SN_FILE):
    grafo_size = len(R)

    social_graph = numpy.zeros((grafo_size, grafo_size))

    social_network = numpy.loadtxt(open(SN_FILE, "rb"), delimiter=",")

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

def load_grafo_social_for_anotherDataSet(R, SN_FILE):
    grafo_size = 2100

    social_graph = numpy.zeros((grafo_size, grafo_size))

    social_network = numpy.loadtxt(open(SN_FILE, "rb"), delimiter='\t')

    for i in xrange(len(social_network)):
        user = int(float(social_network[i][0]))
        friend = int(float(social_network[i][1]))

        if user < grafo_size:
            if friend < grafo_size:

                social_graph[user][friend] = 1
                social_graph[friend][user] = 1

    return social_graph

def findImitates(numNeighbors, alpha):
    imitates = numpy.zeros((len(numNeighbors)+1, len(numNeighbors)+1))
    for i in xrange(len(numNeighbors)):
        sortedAlpha = numpy.argsort(alpha[i, :])[::-1]
        cnt = 0
        while cnt < numNeighbors[i] :

            imitates[i][cnt] = sortedAlpha[cnt]
            cnt = cnt + 1

    return imitates

def findkNN(social_graph, L_C):
    numNeighbors = numpy.zeros(len(social_graph))
    alpha = numpy.zeros((len(social_graph), len(social_graph[1])))
    for i in xrange(len(social_graph)):
        sortIndex = numpy.argsort(social_graph[i])[::-1]
        beta = [L_C * (1 - social_graph[i][j]) for j in sortIndex]
        beta.append(10 ** 6)
        beta.insert(0, 0)
        lamda = beta[1] + 1.0
        k = 0
        Sum_beta = 0
        Sum_beta_square = 0
        now = beta[k + 1]
        lenBe = len(beta)
        while lamda > now and k < lenBe - 1:
            k += 1
            Sum_beta += beta[k]
            Sum_beta_square += (beta[k]) ** 2
            Squrt = numpy.sqrt(k + Sum_beta ** 2 - k * Sum_beta_square)
            lamda = (float)(1.0 / k) * (Sum_beta + Squrt)
        numNeighbors[i] = k - 1

        for j in xrange(len(social_graph[i])):
            temp = lamda - L_C * social_graph[i, j]
            if temp > 0:
                alpha[i, j] = float(temp)
            else:
                alpha[i, j] = 0.0
        summition = numpy.sum(alpha[i, :])
        for j in xrange(len(social_graph[i])):
            if summition != 0:
                alpha[i, j] = alpha[i, j] / summition

    return numNeighbors, alpha

def gd_kNNout(R, U, V, social_graph, steps, stepLength, lamb, betaParam, list_index, imitates, alpha):
    percent = 0
    current_percent = 0
    len_list_index = len(list_index)
    for step in xrange(steps):
        for index in xrange(len_list_index):
            sI, sJ = list_index[index].split(',')

            i = int(float(sI))
            j = int(float(sJ))

            UT = U[i].T
            VT = V[j]
            RT = R[i][j]
            e = numpy.dot(UT, VT) - RT
            u_temp = U[i] - stepLength * (
            (e * V[j]) + (lamb * U[i]) + betaParam * sr_f1(i, U, social_graph, alpha[i, :], imitates[i]))
            V[j] = V[j] - stepLength * ((e * U[i]) + (lamb * V[j]))
            U[i] = u_temp

    return U, V


def gd_kNN_standard(R, U, V, social_graph, steps, stepLength, lamb, betaParam, list_index, imitates, alpha):
    len_list_index = len(list_index)
    for step in xrange(steps):
        for index in xrange(len_list_index):
            sI, sJ = list_index[index].split(',')

            i = int(float(sI))
            j = int(float(sJ))

            UT = U[i].T
            VT = V[j]
            RT = R[i][j]
            e = numpy.dot(UT, VT) - RT
            u_temp = U[i] - stepLength * (
            (e * V[j]) + (lamb * U[i]) + betaParam * sr_f1(i, U, social_graph, alpha[i, :], imitates[i]))
            V[j] = V[j] - stepLength * ((e * U[i]) + (lamb * V[j]))
            U[i] = u_temp

    return U, V


def gd_kNN(R, U, V, social_graph, steps, stepLength, lamb, betaParam, L_C, list_index):
    percent = 0
    current_percent = 0
    len_list_index = len(list_index)
    numNeighbors = numpy.zeros(len(social_graph))
    # alphaTemp = numpy.zeros((len(social_graph), len(social_graph[1])))
    alpha = numpy.zeros((len(social_graph), len(social_graph[1])))
    for i in xrange(len(social_graph)):
        sortIndex = numpy.argsort(social_graph[i])
        beta = [L_C * social_graph[i][j] for j in sortIndex]
        beta.append(10 ** 6)
        beta.insert(0, 0)
        lamda = beta[1] + 1.0
        k = 0
        Sum_beta = 0
        Sum_beta_square = 0
        now = beta[k + 1]
        lenBe = len(beta)
        while lamda > now and k < lenBe - 1:
            k += 1
            Sum_beta += beta[k]
            # print Sum_beta
            Sum_beta_square += (beta[k]) ** 2
            # print Sum_beta_square
            Squrt = numpy.sqrt(k + Sum_beta ** 2 - k * Sum_beta_square)
            # print Squrt
            lamda = (float)(1.0 / k) * (Sum_beta + Squrt)
            # print lamda
        numNeighbors[i] = k - 1
        # if lamda == 0:
        #     numNeighbors[i] = lenBe - 2
        # else:
        #     numNeighbors[i] = k - 1

        for j in xrange(len(social_graph[i])):
            # alpha = lamda - L_C * social_graph
            # alpha[alpha < 0.] = 0.
            temp = lamda - L_C * social_graph[i, j]
            if temp > 0 :
                alpha[i, j] = float(temp)
            else:
                alpha[i, j] = 0.0
            # alphaTemp[i, j] = numpy.max(0, lamda - L_C * social_graph[i, j])
        summition = numpy.sum(alpha[i, :])
        for j in xrange(len(social_graph[i])):
            if summition != 0:
                alpha[i, j] = alpha[i, j]/ summition
        # alpha[i, :] = [numpy.max(0, lamda - L_C * social_graph[i, j]) for j in range(len(social_graph[i]))]
        # alpha[i, :] = alpha[i, :] / numpy.sum(alpha[i, :])

    for step in xrange(steps):
        rmse = 0
        T = 0
        for index in xrange(len_list_index):
            sI, sJ = list_index[index].split(',')

            i = int(float(sI))
            j = int(float(sJ))

            UT = U[i].T
            VT = V[j]
            RT = R[i][j]
            e = numpy.dot(UT, VT) - RT
            u_temp = U[i] - stepLength * (
            (e * V[j]) + (lamb * U[i]) + betaParam * sr_f1(i, U, social_graph, alpha[i, :], numNeighbors[i]))
            V[j] = V[j] - stepLength * ((e * U[i]) + (lamb * V[j]))
            U[i] = u_temp
            T += 1
            rmse += e ** 2

    return U, V, numNeighbors

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


def sr_f1(i, P, SG, weightVector, imitates):
    reg = 0

    for f in imitates:
        reg += SG[i][f] * (P[i] - P[f]) * weightVector[f]

    return reg


def gd_default(R, U, V, social_graph, steps, alpha, lamb, beta, list_index):
    len_list_index = len(list_index)

    for step in xrange(steps):
        for index in xrange(len_list_index):
            sI, sJ = list_index[index].split(',')

            i = int(float(sI))
            j = int(float(sJ))

            e = numpy.dot(U[i].T, V[j]) - R[i][j]
            u_temp = U[i] - alpha * ((e * V[j]) + (lamb * U[i]) + beta * sr_f(i, U, social_graph))
            V[j] = V[j] - alpha * ((e * U[i]) + (lamb * V[j]))
            U[i] = u_temp

    return U, V

