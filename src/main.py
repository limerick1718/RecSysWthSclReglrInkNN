from lib import mtxfac
from lib import mtxfac_sr
from lib import util

import numpy
import time
# import pylab

def exp(exp_name, R, U, V, SN_FILE):
    Bound = [80, 60, 40]
    Step = [300, 1800, 4000]
    Alpha = [0.00002, 0.0002, 0.002]
    Lamb  = [0.0002, 0.002, 0.02]
    Beta  = [0.0001,0.001,0.01]
    L_CRatio = [0.001, 0.01, 0.1]
    for bound in Bound:
        for steps in Step:
            for alpha in Alpha:
                for lamb in Lamb:
                    for beta in Beta:
                        for L_C in L_CRatio:
                            f_result = open('../resultSet/' + exp_name + '_' +'Steps_' + `steps`+ 'Alpha_' + `alpha`+'Lambda_' + `lamb`+ 'Beta_' + `beta`+ 'L_C_' + `L_C`+ 'Bound_' + `bound`, 'w')

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
                        print '******************* FINISH L_C *******************'
                    print '******************* FINISH Beta *******************'
                print '******************* FINISH Lamb *******************'
            print '******************* FINISH Alpha *******************'
        print '******************* FINISH steps *******************'
    print '******************* FINISH *******************'
    f_result.close()

if __name__ == "__main__":

    '************************ EXP GD x GDRS ***************************'

    R = numpy.loadtxt(open("../dataset/NY_MATRIX","rb"),delimiter=",")
    R = numpy.array(R)
    U = numpy.loadtxt(open("../dataset/NY_U","rb"),delimiter=",")
    V = numpy.loadtxt(open("../dataset/NY_V","rb"),delimiter=",")
    SN_FILE = '../dataset/NY_SN'

    exp('NY', R, U, V, SN_FILE)

    R = numpy.loadtxt(open("../dataset/IL_MATRIX","rb"),delimiter=",")
    R = numpy.array(R)
    U = numpy.loadtxt(open("../dataset/IL_U","rb"),delimiter=",")
    V = numpy.loadtxt(open("../dataset/IL_V","rb"),delimiter=",")
    SN_FILE = '../dataset/IL_SN'

    exp('IL', R, U, V, SN_FILE)

    R = numpy.loadtxt(open("../dataset/CA_MATRIX","rb"),delimiter=",")
    R = numpy.array(R)
    U = numpy.loadtxt(open("../dataset/CA_U","rb"),delimiter=",")
    V = numpy.loadtxt(open("../dataset/CA_V","rb"),delimiter=",")
    SN_FILE = '../dataset/CA_SN'

    exp('CA', R, U, V, SN_FILE)
