from lib import mtxfac
from lib import mtxfac_sr
from lib import util

import numpy
import time
import xlwt
# import pylab
import datetime
from numpy import random

def exp(exp_name, R, SN_FILE):
    rowNumber = 0

    Bound = [60]
    Step = [20]
    Alpha = [0.0001]
    Lamb  = [0.01]
    Beta  = [0.1]
    L_CRatio = [0.01,100]

    wb = xlwt.Workbook()
    ws = wb.add_sheet('RssrkNN')
    ws.write(0, 0, 'rowNumber')
    ws.write(0, 1, 'bound')
    ws.write(0, 2, 'step')
    ws.write(0, 3, 'alpha')
    ws.write(0, 4, 'lamb')
    ws.write(0, 5, 'beta')
    ws.write(0, 6, 'L_C')
    ws.write(0, 7, 'SGD_LOSS')
    ws.write(0, 8, 'SGD_time')
    ws.write(0, 9, 'SGD_kNN_LOSS')
    ws.write(0, 10, 'SGD_kNN_time')

    date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    SG = mtxfac_sr.load_grafo_social_for_anotherDataSet(R, SN_FILE)
    U = random.randint(0,0.1,size=(3000, 12000))
    V = random.randint(0,0.1,size=(20000, 12000))

    U1 = numpy.copy(U)
    U2 = numpy.copy(U)
    V1 = numpy.copy(V)
    V2 = numpy.copy(V)

    for bound in Bound:
        # list_index, validation_index = mtxfac.load_matrix_index(R, bound)
        list_index, validation_index, newR = mtxfac.load_matrix_index_for_anotherDataSet(R, bound)
        for step in Step:
            for alpha in Alpha:
                for lamb in Lamb:
                    for beta in Beta:
                        for L_C in L_CRatio:
                            for steps in xrange(step):
                                rowNumber = rowNumber + 1
                                f_result = open('../resultSet/' + exp_name + '_' +'Steps_' + `steps`+ 'Alpha_' + `alpha`+'Lambda_' + `lamb`+ 'Beta_' + `beta`+ 'L_C_' + `L_C`+ 'Bound_' + `bound` + date, 'w')
                                print('bound : ' + `bound` + '\n')
                                print('Steps : ' + `steps` + '\n')
                                print('Alpha : ' + `alpha` + '\n')
                                print('Lambda: ' + `lamb` + '\n')
                                print('Beta  : ' + `beta` + '\n')
                                print('L_C : ' + `L_C` + '\n')
                                f_result.write('bound : ' + `bound` + '\n')
                                f_result.write('Steps : '+ `steps` +'\n')
                                f_result.write('Alpha : '+ `alpha` +'\n')
                                f_result.write('Lambda: '+ `lamb`+'\n')
                                f_result.write('Beta  : '+ `beta`+'\n')
                                f_result.write('L_C : ' + `L_C` + '\n')
                                ws.write(rowNumber, 0, rowNumber)
                                ws.write(rowNumber, 1, bound)
                                ws.write(rowNumber, 2, steps)
                                ws.write(rowNumber, 3, alpha)
                                ws.write(rowNumber, 4, lamb)
                                ws.write(rowNumber, 5, beta)
                                ws.write(rowNumber, 6, L_C)
                                print '******************* SR *******************'
                                print '******************* SGD BEGIN *******************'


                                start_time = time.time()

                                nP1, nQ1 = mtxfac_sr.gd_default(newR, U1, V1, SG, 100, alpha, lamb, beta, list_index)

                                time_exp1 = (time.time() - start_time) / 60

                                exp1 = mtxfac_sr.val(validation_index, newR, nP1, nQ1)

                                f_result.write('SGD: ' + `exp1` + '\n')
                                f_result.write('time  : ' + `time_exp1` + '\n')

                                ws.write(rowNumber, 7, exp1)
                                ws.write(rowNumber, 8, time_exp1)

                                print('SGD: ' + `exp1` + '\n')
                                print('time  : ' + `time_exp1` + '\n')
                                print '******************* FINISH SGD *******************'

                                print '******************* SGD_kNN BEGIN *******************'

                                start_time = time.time()

                                nP2, nQ2, numNeighbors = mtxfac_sr.gd_kNN(newR, U2, V2, SG, 100, alpha, lamb, beta, L_C, list_index)

                                time_exp2 = (time.time() - start_time) / 60

                                exp2=mtxfac_sr.val(validation_index,newR, nP2, nQ2)

                                f_result.write('SGD_kNN: ' + `exp2` + '\n')
                                f_result.write('time  : ' + `time_exp2` + '\n')
                                f_result.write('k :' + `numNeighbors` + '\n')

                                ws.write(rowNumber, 9, exp2)
                                ws.write(rowNumber, 10, time_exp2)
                                # lenOfNeighbor = len(numNeighbors)
                                # for i in xrange(lenOfNeighbor):
                                #     ws.write(rowNumber, 11 + i, numNeighbors[i])

                                print('k :' + `numNeighbors` + '\n')
                                print('SGD_kNN: ' + `exp2` + '\n')
                                print('time  : ' + `time_exp2` + '\n')
                                f_result.close()
                            print '******************* FINISH SGD_kNN *******************'
                        print '******************* FINISH L_C *******************'
                    print '******************* FINISH Beta *******************'
                print '******************* FINISH Lamb *******************'
            print '******************* FINISH Alpha *******************'
        print '******************* FINISH steps *******************'
    wb.save('F:/git/ml/RssrkNN/resultSet/'+ exp_name + date+'.xls')
    print '******************* FINISH *******************'



if __name__ == "__main__":

    '************************ EXP GD x GDRS ***************************'

    R = numpy.loadtxt(open("../dataset/user_artists", "rb"), delimiter='\t')
    R = numpy.array(R)
    # U = numpy.loadtxt(open("../dataset/NY_U", "rb"), delimiter=",")
    # V = numpy.loadtxt(open("../dataset/NY_V", "rb"), delimiter=",")
    SN_FILE = '../dataset/user_friends'

    exp('NY', R, SN_FILE)

    # R = numpy.loadtxt(open("../dataset/NY_MATRIX","rb"),delimiter=",")
    # R = numpy.array(R)
    # U = numpy.loadtxt(open("../dataset/NY_U","rb"),delimiter=",")
    # V = numpy.loadtxt(open("../dataset/NY_V","rb"),delimiter=",")
    # SN_FILE = '../dataset/NY_SN'
    #
    # exp('NY', R, U, V, SN_FILE)
    #
    # R = numpy.loadtxt(open("../dataset/IL_MATRIX","rb"),delimiter=",")
    # R = numpy.array(R)
    # U = numpy.loadtxt(open("../dataset/IL_U","rb"),delimiter=",")
    # V = numpy.loadtxt(open("../dataset/IL_V","rb"),delimiter=",")
    # SN_FILE = '../dataset/IL_SN'
    #
    # exp('IL', R, U, V, SN_FILE)
    #
    # R = numpy.loadtxt(open("../dataset/CA_MATRIX","rb"),delimiter=",")
    # R = numpy.array(R)
    # U = numpy.loadtxt(open("../dataset/CA_U","rb"),delimiter=",")
    # V = numpy.loadtxt(open("../dataset/CA_V","rb"),delimiter=",")
    # SN_FILE = '../dataset/CA_SN'
    #
    # exp('CA', R, U, V, SN_FILE)
