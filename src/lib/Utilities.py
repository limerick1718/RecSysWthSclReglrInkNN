import numpy
from lib import util
from scipy.sparse import csr_matrix
import scipy
import math

def rmse(R, U, V):

    rmse = 0
    step = 0
    row , col = csr_matrix.nonzero(R)
    length = len(row)
    for iterater in xrange(length):
        i = row[iterater]
        j = col[iterater]
        ratingScores = R[i,j]

        step += 1

        UT = U[i]
        VT = V.T[j].T

        e = numpy.dot(UT, VT) - ratingScores
        rmse += e ** 2

    return  numpy.sqrt(rmse/step)


def mae( R, U, V):

    rmse = 0
    step = 0
    row, col = csr_matrix(R).nonzero()
    length = len(row)
    for iterater in xrange(length):
        i = row[iterater]
        j = col[iterater]
        ratingScores = R[i, j]

        step += 1

        UT = U[i]
        VT = V.T[j].T

        e = numpy.dot(UT, VT) - ratingScores
        rmse += abs(e)

    return  rmse/step

def sr_f(i, P, SG):
    reg = 0

    social_relation = csr_matrix(SG).getrow(i)
    friends = social_relation.nonzero()
    for friend in friends:
        reg += SG[i, friend] * (P[i] - P[friend])
    # for j in xrange(len(SG[i])):
    #     reg += SG[i][j][1] * (P[i] - P[SG[i][j][0]])
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

def load_index(bound, R):
    list_index = []
    validation_index = []
    for i in csr_matrix.nonzero(R):
        print i
        randNumber = numpy.random.randint(0, 100)
        if randNumber <= bound:
            list_index.append(`i`)
        else:
            validation_index.append(`i`)

    return list_index, validation_index

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

def load_grafo_social_for_anotherDataSet(R, social_network, userNumber):

    row = []
    col = []
    data = []
    for i in xrange(len(social_network)):
        user = int(float(social_network[i][0]))
        friend = int(float(social_network[i][1]))

        userItemsRating = R.getrow(user).transpose().todense()
        friendsItemsRating = R.getrow(friend).transpose().todense()
        PCC, meiyongde = scipy.stats.pearsonr(userItemsRating, friendsItemsRating)

        row.append(user)
        print len(row)
        col.append(friend)
        print len(col)
        data.append(PCC[0])
        print len(data)

    social_graph_preSim = csr_matrix((data, (row, col)), shape=(userNumber, userNumber))
    print '*******************load social graph success************************'
    return social_graph_preSim

def load_grafo_social_for_anotherDataSet1(newR, social_network, userNumber, itemNumber):
    grafo_size = userNumber

    social_graph = [[] for i in xrange(userNumber + 1)]

    for i in xrange(len(social_network)):
        user = int(float(social_network[i][0]))
        friend = int(float(social_network[i][1]))

        if user <= grafo_size:
            if friend > user:

                social_graph[user].append(friend)
                social_graph[friend].append(user)


    print '*******************load social graph success************************'
    return social_graph

def pearsonr(x, y):

    x = numpy.asarray(x)
    y = numpy.asarray(y)
    n = len(x)
    mx = x.mean()
    my = y.mean()
    xm, ym = x - mx, y - my
    r_num = numpy.add.reduce(xm * ym)
    r_den = numpy.sqrt(scipy.stats._sum_of_squares(xm) * scipy.stats._sum_of_squares(ym))
    r = r_num / r_den


    for i in r:
        i = min(i,1.0)
        i = max(i, -1.0)
    # r = max(min(r, 1.0), -1.0)
    df = n - 2
    if abs(r) == 1.0:
        prob = 0.0
    else:
        t_squared = r**2 * (df / ((1.0 - r) * (1.0 + r)))
        prob = scipy.stats._betai(0.5*df, 0.5, df/(df+t_squared))

    return r, prob

def average(x):
    assert len(x) > 0
    print len(x.T)
    sumation = sum(x)
    print sumation
    return float(sumation) / len(x)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)