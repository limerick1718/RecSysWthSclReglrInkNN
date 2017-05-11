import numpy
from lib import Utilities

def gd_default(R, U, V, social_graph, alpha, lamb, beta, list_index):

    len_list_index = len(list_index)

    round = 0
    Rmse = Utilities.rmse(list_index, R, U, V)
    exitFlag = False
    while True:
        round += 1
        print '**************************************************round*********************************' + `round`
        step = 0
        for index in xrange(len_list_index):
            step += 1
            # print 'step' + `step`
            now = int(float(list_index[index]))
            i = int(float(R[now][0]))
            j = int(float(R[now][1]))
            ratingScores = R[now][2]

            e = numpy.dot(U[i].T, V[j]) - ratingScores
            gdU = (e * V[j]) + (lamb * U[i]) + beta * Utilities.sr_f(i, U, social_graph)
            gdV = (e * U[i]) + (lamb * V[j])
            u_temp = U[i] - alpha * gdU
            V[j] = V[j] - alpha * gdV
            U[i] = u_temp

            Ul2 = numpy.dot(gdU.T, gdU)
            Vl2 = numpy.dot(gdV.T, gdV)

            nowRmse = Utilities.rmse(list_index, R, U, V)
            ratio = (nowRmse * 1.0) / (Rmse * 1.0)
            if round % 100 == 0:
                print round
                print '**********ratio : ' + `ratio - 1`
                print '**********Ul2 : ' + `Ul2`
                print '**********Vl2 : ' + `Vl2`

            if  abs(ratio - 1) < 5 * 10 **(-14):
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


    return U, V, round * len(list_index)+step

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