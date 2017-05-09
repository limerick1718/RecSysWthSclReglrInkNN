import math
import numpy
import datetime
from lib import Utilities
from numpy import random
from random import randint

def userDis(i, f, U, L_C):
    return L_C * numpy.sqrt(numpy.dot((U[i] - U[f]).T, (U[i] - U[f])))

def userDisArray(U, i, L_C, social_graph):
    lenth = len(U)
    disArray = numpy.zeros(lenth)
    for j in xrange(lenth):
        if social_graph[i][j] == 1:
            disArray[j] = userDis(i, j, U, L_C)

    return disArray

def getSoredFriends(disArray):
    return numpy.argsort(disArray)

def RskNN_old(list_index, stepLength, lamb, betaParam, U, V, L_C, R, social_graph):
    len_list_index = len(list_index)
    userNumber = len(U)
    neighbors = numpy.zeros(userNumber)
    Rmse = Utilities.rmse(list_index, R, U, V )
    round = 0
    exitFlag = False
    while True:
        print 'round : ' + `round`
        round += 1
        step = 0
        print datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        for index in xrange(len_list_index):
            # print 'step : ' + `step`
            step += 1
            sI, sJ = list_index[index].split(',')

            i = int(float(sI))
            j = int(float(sJ))

            userArray = userDisArray(U, i, L_C)
            imitates, alpha = findkNN(userArray, U)
            neighbors[i] = len(imitates)
            e = numpy.dot(U[i].T, V[j]) - R[i][j]
            gdU = (e * V[j]) + (lamb * U[i]) + betaParam * Utilities.sr_fkNN(i, U, social_graph, alpha, imitates)
            gdV = (e * U[i]) + (lamb * V[j])

            u_temp = U[i] - stepLength * gdU
            V[j] = V[j] - stepLength * gdV
            U[i] = u_temp

            Ul2 = numpy.dot(gdU.T, gdU)
            Vl2 = numpy.dot(gdV.T, gdV)

            nowRmse = Utilities.rmse(list_index, R, U, V)
            ratio = (nowRmse * 1.0) / (Rmse * 1.0)

            if  abs(ratio - 1) < 5 * 10 ** (-17):
                print '**********final step : ' + `step`
                print '**********ratio : ' + `ratio - 1`
                print '**********Ul2 : ' + `Ul2`
                print '**********Vl2 : ' + `Vl2`
                exitFlag = True
                break
            else:
                Rmse = nowRmse
        if exitFlag == True:
            break

    return U, V, neighbors, round * len(list_index)+step

def RskNN(list_index, stepLength, lamb_phi, lamb_U, lamb_V, beta, U, V, L_C, R, social_graph):
    len_list_index = len(list_index)
    userNumber = len(U)
    neighbors = numpy.zeros(userNumber)
    Rmse = Utilities.rmse(list_index, R, U, V )
    round = 0
    exitFlag = False
    phi = random.uniform(0,0.01,size=(userNumber, 100))
    while True:
        if exitFlag == True:
            break
        # print 'round : ' + `round`
        round += 1
        # step = 0
        # print datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        index = randint(0, len_list_index - 1)

        if exitFlag == True:
            break

        # step += 1
        # print step
        sI, sJ = list_index[index].split(',')

        i = int(float(sI))
        j = int(float(sJ))
        # print "*************************************"
        userArray = userDisArray(U, i, L_C, social_graph)
        imitates, alpha = findkNN(userArray, U)
        neighbors[i] = len(imitates)

        phi_hat = numpy.zeros(len(U[1]))
        for p in imitates:
            phi_hat = alpha[p] * U[p] + phi_hat

        e = numpy.dot(phi[i].T, V[j]) - R[i][j]
        res = phi[i] - phi_hat
        gdPhi = V[j] * e + lamb_phi * res + beta * (phi[i] - U[i])
        gdV = phi[i] * e + lamb_V * V[j]
        gdU = - beta * (phi[i] - U[i]) + lamb_U * U[i]

        U[i] = U[i] - stepLength * gdU
        V[j] = V[j] - stepLength * gdV
        phi[i] = phi[i] - stepLength * gdPhi

        Ul2 = numpy.dot(gdU.T, gdU)
        Vl2 = numpy.dot(gdV.T, gdV)

        nowRmse = Utilities.rmse(list_index, R, U, V)
        ratio = (nowRmse * 1.0) / (Rmse * 1.0)
        if round%100 ==0:
            print round
            print '**********ratio : ' + `ratio - 1`
            print '**********Ul2 : ' + `Ul2`
            print '**********Vl2 : ' + `Vl2`
            print neighbors.nonzero()
            print sum(neighbors[neighbors.nonzero()])/numpy.count_nonzero(neighbors)
        if  abs(ratio - 1) < 5 * 10 ** (-14):
            # print '**********final step : ' + `step`
            print '**********ratio : ' + `ratio - 1`
            print '**********Ul2 : ' + `Ul2`
            print '**********Vl2 : ' + `Vl2`
            exitFlag = True
            break
        else:
            Rmse = nowRmse
        if exitFlag == True:
            break

    return U, V, neighbors, round

def findImitates(numNeighbor, sortedFriends, zeroCount):
    imitates = numpy.zeros(numNeighbor - zeroCount)

    cnt = zeroCount
    number = 0
    while cnt < numNeighbor:
        imitates[number] = sortedFriends[cnt]
        cnt = cnt + 1
        number += 1

    return imitates


def findkNN(userArray, U):
    zeroCount = len(userArray) - numpy.count_nonzero(userArray)
    beta = [userArray[k] for k in xrange(len(U))]
    sortedFriends = getSoredFriends(beta)
    beta.append(10 ** 6)
    lamda = beta[sortedFriends[zeroCount]] + 1.0
    k = zeroCount
    cnt = 0
    Sum_beta = 0
    Sum_beta_square = 0
    while k < len(beta) - 2 and lamda > beta[sortedFriends[k + 1]]:
        k += 1
        cnt += 1
        Sum_beta += beta[k]
        Sum_beta_square += (beta[k]) ** 2
        Squrt = numpy.sqrt(cnt + Sum_beta ** 2 - cnt * Sum_beta_square)
        lamda = (float)(1.0 / cnt) * (Sum_beta + Squrt)

    alpha = numpy.zeros(len(U))
    for p in xrange(len(U)):
        if beta[p] != 0:
            temp = lamda - beta[p]
            if temp > 0:
                alpha[p] = float(temp)
            else:
                alpha[p] = 0.001
    summition = numpy.sum(alpha)
    for p in xrange(len(U)):
            alpha[p] = alpha[p] / summition


    imitates = findImitates(k, sortedFriends, zeroCount)

    return imitates, alpha






