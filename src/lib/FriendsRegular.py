import numpy
from lib import Utilities
from random import randint
from numpy import random
from scipy.sparse import csr_matrix

def alphaGenerator(userNumber, social_graph):
    alpha = numpy.zeros((userNumber, userNumber))
    social_graph = csr_matrix(social_graph)
    for i in xrange(userNumber):
        social_relation = social_graph.getrow(i)
        friendsNumber = social_relation.count_nonzero()
        if friendsNumber != 0:
            weight = 1.0/friendsNumber

            for j in social_relation.nonzero():
                alpha[i,j] = weight
    alpha = csr_matrix(alpha)
    return alpha


def FR(R, stepLength, lamb_phi, lamb_U, lamb_V, lamb_alpha, U, V, social_graph, userNumber):
    row, col = csr_matrix(R).nonzero()
    length = len(row)
    phi = random.uniform(0, 0.01, size=(userNumber, 10))
    round = 0
    Rmse = Utilities.rmse(R, U, V)
    exitFlag = False
    alpha = alphaGenerator(userNumber, social_graph)
    while True:
        if exitFlag == True:
            break
        round += 1
        index = randint(0, length - 1)
        i = row[index]
        j = col[index]
        ratingScores = R[i,j]

        phi_hat = numpy.zeros(len(U[1]))
        for p in social_graph.getrow(i).nonzero():
            phi_hat = alpha[i,p] * U[p] + phi_hat

        e = numpy.dot(phi[i], V.T[j].T) - ratingScores
        res = phi[i] - phi_hat
        gdPhi = V.T[j].T * e + lamb_phi * res
        gdV = phi[i] * e + lamb_V * V.T[j].T
        gdU = - lamb_phi * res * alpha[i,i] + lamb_U * U[i]
        Selse = numpy.zeros(len(U))
        for p in social_graph.getrow(i).nonzero():
            left = numpy.dot(U[p] ,(phi[i] - alpha[i,p] * U[p]).T)
            Selse[p] = left + Selse[p]
        gdAlpha = - lamb_phi * Selse + lamb_alpha * alpha[i]

        U[i] = U[i] - stepLength * gdU
        V.T[j] = (V.T[j].T - stepLength * gdV).T
        phi[i] = phi[i] - stepLength * gdPhi
        alpha[i] = alpha[i] - stepLength * gdAlpha

        Ul2 = numpy.dot(gdU, gdU.T)
        Vl2 = numpy.dot(gdV.T, gdV)

        nowRmse = Utilities.rmse( R, U, V)
        ratio = (nowRmse * 1.0) / (Rmse * 1.0)
        if round % 30 == 0:
            print round
            print '**********ratio : ' + `ratio - 1`
            print '**********Ul2 : ' + `Ul2`
            print '**********Vl2 : ' + `Vl2`
        if abs(ratio - 1) < 5 * 10 ** (-7) :#and Ul2[0][0] < 10**(-14) and Vl2 < 10 **(-10):
            print '**********ratio : ' + `ratio - 1`
            print '**********Ul2 : ' + `Ul2`
            print '**********Vl2 : ' + `Vl2`
            exitFlag = True
            break
        else:
            Rmse = nowRmse

    return U, V, round