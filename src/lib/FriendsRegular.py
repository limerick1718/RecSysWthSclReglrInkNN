import numpy
import datetime
from lib import Utilities
from random import randint
from numpy import random

def alphaGenerator(userNumber, social_graph):
    alpha = numpy.zeros((userNumber, userNumber))
    for i in xrange(userNumber):
        friendsNumber = len(social_graph[i])
        if friendsNumber != 0:
            weight = 1.0/friendsNumber

            for j in social_graph[i]:
                alpha[i][j] = weight

    return alpha


def FR(list_index, stepLength, lamb_phi, lamb_U, lamb_V, lamb_alpha, U, V, R, social_graph):
    len_list_index = len(list_index)
    userNumber = len(U)
    phi = random.uniform(0, 0.01, size=(userNumber, 100))
    round = 0
    Rmse = Utilities.rmse(list_index, R, U, V)
    exitFlag = False
    alpha = alphaGenerator(userNumber, social_graph)
    while True:
        if exitFlag == True:
            break
        round += 1
        index = randint(0, len_list_index - 1)
        now = int(float(list_index[index]))
        i = int(float(R[now][0]))
        j = int(float(R[now][1]))
        ratingScores = R[now][2]

        phi_hat = numpy.zeros(len(V))
        for p in social_graph[i]:
            phi_hat = alpha[i][p] * U[p] + phi_hat

        e = numpy.dot(phi[i].T, V[j]) - ratingScores
        res = phi[i] - phi_hat
        gdPhi = V[j] * e + lamb_phi * res
        gdV = phi[i] * e + lamb_V * V[j]
        gdU = - lamb_phi * res * alpha[i][i] + lamb_U * U[i]
        Selse = numpy.zeros(len(U))
        for p in social_graph[i]:
            left = numpy.dot(U[p] ,(phi[i] - alpha[i][p] * U[p]))
            Selse[p] = left + Selse[p]
        gdAlpha = - lamb_phi * Selse + lamb_alpha * alpha[i]

        U[i] = U[i] - stepLength * gdU
        V[j] = V[j] - stepLength * gdV
        phi[i] = phi[i] - stepLength * gdPhi
        alpha[i] = alpha[i] - stepLength * gdAlpha

        Ul2 = numpy.dot(gdU.T, gdU)
        Vl2 = numpy.dot(gdV.T, gdV)

        nowRmse = Utilities.rmse(list_index, R, U, V)
        ratio = (nowRmse * 1.0) / (Rmse * 1.0)
        if round % 100 == 0:
            print round
            print '**********ratio : ' + `ratio - 1`
            print '**********Ul2 : ' + `Ul2`
            print '**********Vl2 : ' + `Vl2`
        if abs(ratio - 1) < 5 * 10 ** (-14):
            print '**********ratio : ' + `ratio - 1`
            print '**********Ul2 : ' + `Ul2`
            print '**********Vl2 : ' + `Vl2`
            exitFlag = True
            break
        else:
            Rmse = nowRmse

    return U, V, round