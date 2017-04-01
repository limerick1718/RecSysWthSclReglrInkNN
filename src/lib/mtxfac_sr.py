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
        user = social_network[i][0]
        friend = social_network[i][1]

        x = R[user]
        y = R[friend]

        cor_pearson = util.pearson(x, y)

        social_graph[user][friend] = cor_pearson
        social_graph[friend][user] = cor_pearson

    # print social_graph

    return social_graph


def load_grafo_social_for_anotherDataSet(R, SN_FILE):
    grafo_size = 2200

    social_graph = numpy.zeros((2200, 2200))

    social_network = numpy.loadtxt(open(SN_FILE, "rb"), delimiter='\t')

    for i in xrange(len(social_network)):
        user = float(social_network[i][0])
        friend = float(social_network[i][1])

        x = R[user]
        y = R[friend]

        cor_pearson = util.pearson(x, y)

        social_graph[user][friend] = cor_pearson
        social_graph[friend][user] = cor_pearson

    # print social_graph

    return social_graph

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
        lamda = beta[0] + 1
        k = 0
        Sum_beta = 0
        Sum_beta_square = 0
        while lamda > beta[k + 1] and k < len(beta) - 1:
            k += 1
            Sum_beta += beta[k]
            Sum_beta_square += (beta[k]) ** 2
            lamda = (1 / k) * (Sum_beta + numpy.sqrt(k + Sum_beta ** 2 - k * Sum_beta_square))
        numNeighbors[i] = k

        for j in xrange(len(social_graph[i])):
            alpha = lamda - L_C * social_graph
            alpha[alpha < 0.] = 0.
            # temp = lamda - L_C * social_graph[i, j]
            # if temp > 0 :
            #     alpha[i, j] = float(temp)
            # else:
            #     alpha[i, j] = 0.0
            # alphaTemp[i, j] = numpy.max(0, lamda - L_C * social_graph[i, j])
        summition = numpy.sum(alpha[i, :])
        for j in xrange(len(social_graph[i])):
            alpha[i, j] = alpha[i, j]/ summition
        # alpha[i, :] = [numpy.max(0, lamda - L_C * social_graph[i, j]) for j in range(len(social_graph[i]))]
        # alpha[i, :] = alpha[i, :] / numpy.sum(alpha[i, :])

    for step in xrange(steps):

        for index in xrange(len_list_index):
            sI, sJ = list_index[index].split(',')

            i = int(float(sI))
            j = int(float(sJ))

            UT = U[i]
            VT = V[j].T
            RT = R[i][j]
            e = numpy.dot(UT, VT) - RT
            u_temp = U[i] - stepLength * (
            (e * V[j]) + (lamb * U[i]) + betaParam * sr_f1(i, U, social_graph, alpha[i, :], numNeighbors[i]))
            V[j] = V[j] - stepLength * ((e * U[i]) + (lamb * V[j]))
            U[i] = u_temp

        current_percent = util.calc_progress(steps, step + 1, current_percent)

        if (current_percent != percent):
            print current_percent
            percent = current_percent

    return U, V, numNeighbors

def sgd_kNN(R, U, V, SN_FILE, steps, stepLength, lamb, betaParam, L_C, list_index):
    percent = 0
    current_percent = 0
    len_list_index=len(list_index)
    social_graph = load_grafo_social(R, SN_FILE)
    numNeighbors = numpy.zeros(len(social_graph))
    alpha = numpy.zeros((len(social_graph), len(social_graph[1])))
    for i in xrange(len(social_graph)):
        sortIndex = numpy.argsort(social_graph[i])
        beta = [L_C * social_graph[i][j] for j in sortIndex]
        beta.append(10 ** 6)
        beta.insert(0, 0)
        lamda = beta[0] + 1
        k = 0
        Sum_beta = 0
        Sum_beta_square = 0
        while lamda > beta[k + 1] and k < len(beta) - 1:
            k = k + 1
            Sum_beta = Sum_beta + beta[k]
            Sum_beta_square = Sum_beta_square + (beta[k]) ** 2
            lamda = (1 / k) * (Sum_beta + numpy.sqrt(k + Sum_beta ** 2 - k * Sum_beta_square))
        numNeighbors[i] = k

        alpha[i, :] = [numpy.max(0, lamda - L_C * social_graph[i, j]) for j in range(len(social_graph[i]))]
        alpha[i, :] = alpha[i, :] / numpy.sum(alpha[i, :])
    for step in xrange(steps):

        index = randint(0,len_list_index-1)

        sI,sJ =  list_index[index].split(',')

        i = int(sI)
        j = int(sJ)

        e = numpy.dot(U[i].T, V[j]) - R[i][j]
        u_temp = U[i] - stepLength * (
            (e * V[j]) + (lamb * U[i]) + betaParam * sr_f1(i, U, social_graph, alpha[i, :], numNeighbors[i]))
        V[j] = V[j] - stepLength * ((e * U[i]) + (lamb * V[j]))
        U[i] = u_temp

        current_percent = util.calc_progress(steps, step+1, current_percent)

        if (current_percent != percent):
            # print current_percent
            percent = current_percent

    return U, V, numNeighbors

def val(validation_index, R, U, V):

    rmse = 0
    T = 0
    for index in xrange(len(validation_index)):
        sI,sJ =  validation_index[index].split(',')

        i = int(float(sI))
        j = int(float(sI))

        T += 1

        UT = U[i]
        VT = V[j].T
        RT = R[i][j]
        e = numpy.dot(UT, VT) - RT
        rmse += e ** 2

    return  numpy.sqrt(rmse/T)

def sgd(R, U, V, SN_FILE, steps, stepLength, lamb, betaParam, list_index):
    percent = 0
    current_percent = 0
    len_list_index=len(list_index)
    social_graph = load_grafo_social(R, SN_FILE)
    for step in xrange(steps):

        index = randint(0,len_list_index-1)
        sI,sJ =  list_index[index].split(',')

        i = int(sI)
        j = int(sJ)

        e = numpy.dot(U[i].T, V[j]) - R[i][j]
        u_temp = U[i] - stepLength * (
            (e * V[j]) + (lamb * U[i]) + betaParam * sr_f(i, U, social_graph))
        V[j] = V[j] - stepLength * ((e * U[i]) + (lamb * V[j]))
        U[i] = u_temp

        current_percent = util.calc_progress(steps, step+1, current_percent)

        if (current_percent != percent):
            # print current_percent
            percent = current_percent

    return U, V

def sr_f(i, P, SG):
    reg = 0

    for f in xrange(len(SG[i])):
        if SG[i][f] > 0:
            reg += SG[i][f] * (P[i] - P[f])

    return reg


def sr_f1(i, P, SG, weightVector, k):
    reg = 0
    sortIndex = numpy.argsort(SG[i])

    for f in xrange(int(k)):
        if SG[i][sortIndex[f]] > 0:
            if ~isnan(weightVector[sortIndex[f]]):
                reg += SG[i][sortIndex[f]] * (P[i] - P[sortIndex[f]]) * weightVector[sortIndex[f]]

    return reg


def gd_default(R, U, V, social_graph, steps, alpha, lamb, beta, list_index):
    percent = 0
    current_percent = 0

    len_list_index = len(list_index)

    for step in xrange(steps):

        for index in xrange(len_list_index):
            sI, sJ = list_index[index].split(',')

            i = int(float(sI))
            j = int(float(sJ))

            e = numpy.dot(U[i], V[j].T) - R[i][j]
            u_temp = U[i] - alpha * ((e * V[j]) + (lamb * U[i]) + beta * sr_f(i, U, social_graph))
            V[j] = V[j] - alpha * ((e * U[i]) + (lamb * V[j]))
            U[i] = u_temp

        current_percent = util.calc_progress(steps, step + 1, current_percent)

        if (current_percent != percent):
            print current_percent
            percent = current_percent

    return U, V


def gd_distributed(R, U, V, steps, alpha, lamb, beta, SG, FULL_U, REAL_INDEX):
    list_index = mtxfac.load_matrix_index(R)

    len_list_index = len(list_index)

    for step in xrange(steps):

        for index in xrange(len(list_index)):
            sI, sJ = list_index[index].split(',')

            i = int(sI)
            j = int(sJ)

            e = numpy.dot(U[i].T, V[j]) - R[i][j]
            u_temp = U[i] - alpha * ((e * V[j]) + (lamb * U[i]) + beta * sr_f(REAL_INDEX + i, FULL_U, SG))
            V[j] = V[j] - alpha * ((e * U[i]) + (lamb * V[j]))
            U[i] = u_temp

    return U, V


def dgd(R, U, V, stratus_number, T, steps, alpha, lamb, beta):
    percent = 0
    current_percent = 0

    SG = load_grafo_social(R)

    for step in xrange(T):

        list_stratus, list_U, list_V, index_pointer_r, index_pointer_c = stratus.split_matrix(R, U, V, stratus_number,
                                                                                              step)

        for i in xrange(stratus_number):
            list_U[i], list_V[i] = gd_distributed(list_stratus[i], list_U[i], list_V[i], steps, alpha, lamb, beta, SG,
                                                  U, index_pointer_r[i])

        index_U = 0
        for index_array in xrange(stratus_number):
            temp_U = list_U[index_array]

            for i in xrange(len(temp_U)):
                for j in xrange(len(temp_U[0])):
                    U[index_U][j] = temp_U[i][j]
                index_U += 1

        index_V = 0
        for x in xrange(stratus_number):
            index_V = index_pointer_c[x]

            temp_V = list_V[x]

            for i in xrange(len(temp_V)):
                V[index_V + i] = temp_V[i]

        current_percent = util.calc_progress(T, step + 1, current_percent)

        if (current_percent != percent):
            print current_percent
            percent = current_percent

    return U, V