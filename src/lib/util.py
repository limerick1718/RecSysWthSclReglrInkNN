import numpy
import math

def rmse(predictions, R):
    rmse = 0
    T = 0
    for i in xrange(len(R)):
        for j in xrange(len(R[i])):
            if R[i][j] > 0:
                T += 1
                rmse += (predictions[i][j] - R[i][j]) ** 2

    return  numpy.sqrt(rmse/T)

def pearson(x, y):
    assert len(x) > 0
    assert len(x) == len(y)

    dense_X = []
    dense_Y = []

    for i in xrange(len(x)):
        if x[i] > 0 and y[i] > 0:
            dense_X.append(x[i])
            dense_Y.append(y[i])

    
    n = len(dense_X)

    if n == 0:
        return 0

    avg_x = float(sum(dense_X)) / len(dense_X)
    avg_y = float(sum(dense_Y)) / len(dense_Y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = dense_X[idx] - avg_x
        ydiff = dense_Y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    sim = diffprod / math.sqrt(xdiff2 * ydiff2)

    if math.isnan(sim):
        sim = 0

    # return (sim+1)/2
    return sim

def calc_progress(total, current_step, current_percent):

    percent = current_step*100/total

    if percent != current_percent: #and percent % 5 == 0:
        return percent
    else:
        return current_percent

def generate_U_V(M,N,K, u_file, v_file):

    U = numpy.random.rand(M,K)
    V = numpy.random.rand(N,K)

    f_U = open('../dataset/'+u_file, 'w')

    for i in xrange(len(U)):

        line = ''

        for j in xrange(len(U[0])):

            line += `U[i][j]`

            if j+1 < len(U[0]):
                line += ','
            else:
                line += '\n'

        f_U.write(line)

    f_U.close()

    f_V = open('../dataset/'+v_file, 'w')

    for i in xrange(len(V)):

        line = ''

        for j in xrange(len(V[0])):

            line += `V[i][j]`

            if j+1 < len(V[0]):
                line += ','
            else:
                line += '\n'

        f_V.write(line)

    f_V.close()