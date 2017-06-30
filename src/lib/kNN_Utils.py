import math
import numpy
import datetime
from lib import Utilities
from numpy import random
from random import randint
from scipy.sparse import csr_matrix

global userNumber
def userDis(i, f, U, L_C):
    return L_C * numpy.sqrt(numpy.dot((U[i] - U[f]).T, (U[i] - U[f])))

def userDisArray(U, i, L_C, social_graph):
    global userNumber
    # print i
    disArray = numpy.zeros(userNumber)
    social_relation = csr_matrix(social_graph).getrow(i)
    row, col = social_relation.nonzero()
    for j in xrange(userNumber):
        if j in col:
            # print j
            # disArray[int(j)] = int(10 ** 6)
            disArray[j] = userDis(i, j, U, L_C)
        else:
            disArray[int(j)] = int(10 ** 6)
            # disArray[j] = userDis(i, j, U, L_C)

    return disArray, social_relation.count_nonzero()

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

def RskNN(R, stepLength, lamb_phi, lamb_U, lamb_V, beta, U, V, L_C, social_graph, userCount):
    global userNumber
    userNumber = userCount
    row, col = csr_matrix(R).nonzero()
    length = len(row)
    neighbors = numpy.zeros(userNumber)
    Rmse = Utilities.rmse(R, U, V )
    round = 0
    exitFlag = False
    phi = random.uniform(0,0.01,size=(userNumber, 10))
    while True:
        if exitFlag == True:
            break
        round += 1
        index = randint(0, length - 1)
        if exitFlag == True:
            break
        i = row[index]
        j = col[index]
        ratingScores = R[i,j]

        userArray, friendNumber = userDisArray(U, i, L_C, social_graph)
        # imitates, alpha = findkNN(userArray, U)
        # imitates, alpha = findkNN_soft(userArray, U, 0.01)
        imitates, alpha = findingkNN(userArray, U, friendNumber)
        neighbors[i] = len(imitates)

        phi_hat = numpy.zeros(len(U[1]))
        for p in imitates:
            phi_hat = alpha[p] * U[p] + phi_hat

        e = numpy.dot(phi[i], V.T[j].T) - ratingScores
        res = phi[i] - phi_hat
        gdPhi = V.T[j].T * e + lamb_phi * res + beta * (phi[i] - U[i])
        gdV = phi[i] * e + lamb_V * V.T[j].T
        gdU = - beta * (phi[i] - U[i]) + lamb_U * U[i]

        U[i] = U[i] - stepLength * gdU
        V.T[j] = (V.T[j].T - stepLength * gdV).T
        phi[i] = phi[i] - stepLength * gdPhi

        Ul2 = numpy.dot(gdU, gdU.T)
        Vl2 = numpy.dot(gdV.T, gdV)

        nowRmse = Utilities.rmse( R, U, V)
        ratio = (nowRmse * 1.0) / (Rmse * 1.0)
        if round%30 ==0:
            print round
            print '**********ratio : ' + `ratio - 1`
            print '**********Ul2 : ' + `Ul2`
            print '**********Vl2 : ' + `Vl2`
            print neighbors.nonzero()
            # print sum(neighbors[neighbors.nonzero()])/numpy.count_nonzero(neighbors)
        if  abs(ratio - 1) < 5 * 10 ** (-7):
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


def findingImitates(numNeighbor, sortedFriends):
    imitates = numpy.zeros(numNeighbor)

    number = 0
    while number < numNeighbor:
        imitates[number] = sortedFriends[number]
        number += 1

    return imitates

def findImitates(numNeighbor, sortedFriends, zeroCount):
    imitates = numpy.zeros(numNeighbor - zeroCount)

    cnt = zeroCount
    number = 0
    while cnt < numNeighbor:
        imitates[number] = sortedFriends[cnt]
        cnt = cnt + 1
        number += 1

    return imitates

def findingkNN(userArray, U, friendNumber):
    global userNumber
    beta = [userArray[k] for k in xrange(userNumber)]
    sortedFriends = getSoredFriends(beta)
    beta.append(10 ** 6)
    lamda = beta[sortedFriends[0]] + 1.0
    k = 0
    Sum_beta = 0
    Sum_beta_square = 0
    while k < len(beta) - 2 and k < friendNumber and lamda > beta[sortedFriends[k + 1]]:
        Sum_beta += beta[sortedFriends[k]]
        Sum_beta_square += (beta[sortedFriends[k]]) ** 2
        k += 1
        Squrt = numpy.sqrt(k + Sum_beta ** 2 - k * Sum_beta_square)
        lamda = (float)(1.0 / k) * (Sum_beta + Squrt)

    alpha = numpy.zeros(userNumber)

    cnt = 0
    while cnt < k:
        temp = lamda - beta[sortedFriends[cnt]]
        if temp > 0:
            alpha[sortedFriends[cnt]] = float(temp)
        else:
            alpha[sortedFriends[cnt]] = 0
        cnt+=1
    summition = numpy.sum(alpha)
    if summition == 0:
        summition = 10**(-15)
    for p in xrange(len(U)):
            alpha[p] = float(alpha[p]) / float(summition)

    imitates = findingImitates(k, sortedFriends)

    return imitates, alpha


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


def findkNN_soft(userArray, U, theta):
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
                alpha[p] = math.exp(- theta * float(temp))
            else:
                alpha[p] = 0
    summition = numpy.sum(alpha)
    for p in xrange(len(U)):
        if summition != 0:
            alpha[p] = alpha[p] / summition
        else:
            alpha[p] = 1 / len(U)

    imitates = findImitates(k, sortedFriends, zeroCount)

    return imitates, alpha






