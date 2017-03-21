from lib import mtxfac
from lib import mtxfac_sr
from lib import util

import numpy
import time
# import pylab

def exp(exp_name, R, U, V, SN_FILE, steps, bound):

    alpha = 0.0002
    lamb  = 0.002
    beta  = 0.001
    L_C = 0.1
    f_result = open('../resultSet/' + exp_name + '_' +'Steps : ' + `steps`+ 'Alpha : ' + `alpha`+'Lambda: ' + `lamb`+ 'Beta  : ' + `beta`+ 'L_C  : ' + `L_C`, 'w')

    print('Steps : ' + `steps` + '\n')
    print('Alpha : ' + `alpha` + '\n')
    print('Lambda: ' + `lamb` + '\n')
    print('Beta  : ' + `beta` + '\n')
    f_result.write('Steps : '+`steps`+'\n')
    f_result.write('Alpha : '+`alpha`+'\n')
    f_result.write('Lambda: '+`lamb`+'\n')
    f_result.write('Beta  : '+`beta`+'\n')
    print '******************* SR *******************'
    print '******************* SGD BEGIN *******************'

    U1 = numpy.copy(U)
    V1 = numpy.copy(V)
    list_index, validation_index = mtxfac.load_matrix_index(R, bound)
    start_time = time.time()

    nP1, nQ1 = mtxfac_sr.sgd(R, U, V, SN_FILE, steps, alpha, lamb, beta, list_index)
    exp1 = mtxfac_sr.val(validation_index, R, nP1, nQ1)

    time_exp1 = (time.time() - start_time) / 60

    f_result.write('SGD: ' + `exp1` + '\n')
    f_result.write('time  : ' + `time_exp1` + '\n')

    print('SGD: ' + `exp1` + '\n')
    print('time  : ' + `time_exp1` + '\n')
    print '******************* FINISH SGD *******************'

    print '******************* SGD_kNN BEGIN *******************'

    U2 = numpy.copy(U)
    V2 = numpy.copy(V)

    start_time = time.time()

    list_index, validation_index = mtxfac.load_matrix_index(R, bound)

    nP2, nQ2, numNeighbors = mtxfac_sr.sgd_kNN(R, U2, V2, SN_FILE, steps, alpha, lamb, beta, L_C, list_index)
    exp2=mtxfac_sr.val(validation_index,R, nP2, nQ2)

    time_exp2 = (time.time() - start_time) / 60

    f_result.write('SGD: ' + `exp2` + '\n')
    f_result.write('time  : ' + `time_exp2` + '\n')
    f_result.write('k :' + `numNeighbors` + '\n')

    print('k :' + `numNeighbors` + '\n')
    print('SGD: ' + `exp2` + '\n')
    print('time  : ' + `time_exp2` + '\n')
    print '******************* FINISH SGD_kNN *******************'
    print '******************* FINISH *******************'
    f_result.close()

if __name__ == "__main__":

    '************************ EXP GD x GDRS ***************************'

    R = numpy.loadtxt(open("../dataset/NY_MATRIX","rb"),delimiter=",")
    R = numpy.array(R)
    U = numpy.loadtxt(open("../dataset/NY_U","rb"),delimiter=",")
    V = numpy.loadtxt(open("../dataset/NY_V","rb"),delimiter=",")
    SN_FILE = '../dataset/NY_SN'

    exp('NY', R, U, V, SN_FILE, 1800000, 80)

    R = numpy.loadtxt(open("../dataset/IL_MATRIX","rb"),delimiter=",")
    R = numpy.array(R)
    U = numpy.loadtxt(open("../dataset/IL_U","rb"),delimiter=",")
    V = numpy.loadtxt(open("../dataset/IL_V","rb"),delimiter=",")
    SN_FILE = '../dataset/IL_SN'

    exp('IL', R, U, V, SN_FILE, 1800000, 80 )

    R = numpy.loadtxt(open("../dataset/CA_MATRIX","rb"),delimiter=",")
    R = numpy.array(R)
    U = numpy.loadtxt(open("../dataset/CA_U","rb"),delimiter=",")
    V = numpy.loadtxt(open("../dataset/CA_V","rb"),delimiter=",")
    SN_FILE = '../dataset/CA_SN'

    exp('CA', R, U, V, SN_FILE, 1800000, 80)
