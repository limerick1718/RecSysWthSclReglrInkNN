from lib import mtxfac
from lib import mtxfac_sr
from lib import util

import numpy
import time
import xlwt
# import pylab
import datetime
from numpy import random

def exp(exp_name, R, U, V, SN_FILE, isTraditionalFile):
    rowNumber = 0

    Bound = [60]
    Step = [10]
    Alpha = [0.001]
    Lamb  = [0.01]
    Beta  = [0.001]
    L_CRatio = [10]

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
    if isTraditionalFile:
        SG = mtxfac_sr.load_grafo_social(R, SN_FILE)
    else:
        SG = mtxfac_sr.load_grafo_social_for_anotherDataSet(R, SN_FILE)


    U1 = numpy.copy(U)
    # U2 = numpy.copy(U)
    U3 = numpy.copy(U)
    # U4 = numpy.copy(U)
    U5 = numpy.copy(U)
    # U6 = numpy.copy(U)
    U7 = numpy.copy(U)
    # U8 = numpy.copy(U)
    U9 = numpy.copy(U)
    U10 = numpy.copy(U)
    V1 = numpy.copy(V)
    # V2 = numpy.copy(V)
    V3 = numpy.copy(V)
    # V4 = numpy.copy(V)
    V5 = numpy.copy(V)
    # V6 = numpy.copy(V)
    V7 = numpy.copy(V)
    # V8 = numpy.copy(V)
    V9 = numpy.copy(V)
    V10 = numpy.copy(V)

    for bound in Bound:
        if isTraditionalFile:
            list_index, validation_index = mtxfac.load_matrix_index(R, bound)
        else:
            list_index, validation_index, R = mtxfac.load_matrix_index_for_anotherDataSet(R, bound)
        for step in Step:
            for alpha in Alpha:
                for lamb in Lamb:
                    for beta in Beta:
                        for steps in xrange(step):
                            rowNumber = rowNumber + 1
                            f_result = open('../resultSet/' + exp_name + '_' +'Steps_' + `steps`+ 'Alpha_' + `alpha`+'Lambda_' + `lamb`+ 'Beta_' + `beta`+ 'Bound_' + `bound` + date, 'w')
                            print('bound : ' + `bound` + '\n')
                            print('Steps : ' + `steps` + '\n')
                            print('Alpha : ' + `alpha` + '\n')
                            print('Lambda: ' + `lamb` + '\n')
                            print('Beta  : ' + `beta` + '\n')
                            # print('L_C : ' + `L_C` + '\n')
                            f_result.write('bound : ' + `bound` + '\n')
                            f_result.write('Steps : '+ `steps` +'\n')
                            f_result.write('Alpha : '+ `alpha` +'\n')
                            f_result.write('Lambda: '+ `lamb`+'\n')
                            f_result.write('Beta  : '+ `beta`+'\n')
                            # f_result.write('L_C : ' + `L_C` + '\n')
                            ws.write(rowNumber, 0, rowNumber)
                            ws.write(rowNumber, 1, bound)
                            ws.write(rowNumber, 2, steps)
                            ws.write(rowNumber, 3, alpha)
                            ws.write(rowNumber, 4, lamb)
                            ws.write(rowNumber, 5, beta)
                            # ws.write(rowNumber, 6, L_C)
                            print '******************* SR *******************'
                            print '******************* SGD BEGIN *******************'

                            start_time = time.time()

                            nP1, nQ1 = mtxfac_sr.gd_default(R, U1, V1, SG, 50, alpha, lamb, beta, list_index)

                            time_exp1 = (time.time() - start_time) / 60

                            exp1 = mtxfac_sr.val(validation_index, R, nP1, nQ1)

                            f_result.write('SGD: ' + `exp1` + '\n')
                            f_result.write('time  : ' + `time_exp1` + '\n')

                            ws.write(rowNumber, 7, exp1)
                            ws.write(rowNumber, 8, time_exp1)

                            print('SGD: ' + `exp1` + '\n')
                            print('time  : ' + `time_exp1` + '\n')
                            print '******************* FINISH SGD *******************'

                            # print '******************* SGD_kNN BEGIN *******************'
                            # L_C = 0.001
                            # f_result.write('L_C : ' + `L_C` + '\n')
                            #
                            # rowNumber = rowNumber + 1
                            # ws.write(rowNumber, 0, rowNumber)
                            # ws.write(rowNumber, 1, bound)
                            # ws.write(rowNumber, 2, steps)
                            # ws.write(rowNumber, 3, alpha)
                            # ws.write(rowNumber, 4, lamb)
                            # ws.write(rowNumber, 5, beta)
                            #
                            # ws.write(rowNumber, 6, L_C)
                            # start_time = time.time()
                            #
                            # nP2, nQ2, numNeighbors = mtxfac_sr.gd_kNN(newR, U2, V2, SG, 100, alpha, lamb, beta, L_C, list_index)
                            #
                            # time_exp2 = (time.time() - start_time) / 60
                            #
                            # exp2=mtxfac_sr.val(validation_index,newR, nP2, nQ2)
                            #
                            # f_result.write('SGD_kNN: ' + `exp2` + '\n')
                            # f_result.write('time  : ' + `time_exp2` + '\n')
                            # f_result.write('k :' + `numNeighbors` + '\n')
                            #
                            # ws.write(rowNumber, 9, exp2)
                            # ws.write(rowNumber, 10, time_exp2)
                            # # lenOfNeighbor = len(numNeighbors)
                            # # for i in xrange(lenOfNeighbor):
                            # #     ws.write(rowNumber, 11 + i, numNeighbors[i])
                            #
                            # print('k :' + `numNeighbors` + '\n')
                            # print('SGD_kNN: ' + `exp2` + '\n')
                            # print('time  : ' + `time_exp2` + '\n')
                            # print '******************* FINISH SGD_kNN *******************'

                            print '******************* SGD_kNN BEGIN *******************'
                            L_C = 0.5
                            f_result.write('L_C : ' + `L_C` + '\n')
                            rowNumber = rowNumber + 1
                            ws.write(rowNumber, 0, rowNumber)
                            ws.write(rowNumber, 1, bound)
                            ws.write(rowNumber, 2, steps)
                            ws.write(rowNumber, 3, alpha)
                            ws.write(rowNumber, 4, lamb)
                            ws.write(rowNumber, 5, beta)
                            ws.write(rowNumber, 6, L_C)
                            weightVector, numNeighbors = mtxfac_sr.kNN(SG, L_C)
                            start_time = time.time()

                            nP3, nQ3, numNeighbors = mtxfac_sr.gd_kNN(R, U3, V3, SG, 50, alpha, lamb, beta, numNeighbors,
                                                                      list_index, weightVector)

                            time_exp3 = (time.time() - start_time) / 60

                            exp3=mtxfac_sr.val(validation_index,R, nP3, nQ3)

                            f_result.write('SGD_kNN: ' + `exp3` + '\n')
                            f_result.write('time  : ' + `time_exp3` + '\n')
                            f_result.write('k :' + `numNeighbors` + '\n')

                            ws.write(rowNumber, 9, exp3)
                            ws.write(rowNumber, 10, time_exp3)
                            # lenOfNeighbor = len(numNeighbors)
                            # for i in xrange(lenOfNeighbor):
                            #     ws.write(rowNumber, 11 + i, numNeighbors[i])

                            print('k :' + `numNeighbors` + '\n')
                            print('SGD_kNN: ' + `exp3` + '\n')
                            print('time  : ' + `time_exp3` + '\n')
                            print '******************* FINISH SGD_kNN *******************'
                            #
                            # # print '******************* SGD_kNN BEGIN *******************'
                            # # L_C = 0.01
                            # # f_result.write('L_C : ' + `L_C` + '\n')
                            # # rowNumber = rowNumber + 1
                            # # ws.write(rowNumber, 0, rowNumber)
                            # # ws.write(rowNumber, 1, bound)
                            # # ws.write(rowNumber, 2, steps)
                            # # ws.write(rowNumber, 3, alpha)
                            # # ws.write(rowNumber, 4, lamb)
                            # # ws.write(rowNumber, 5, beta)
                            # # ws.write(rowNumber, 6, L_C)
                            # # start_time = time.time()
                            # #
                            # # nP4, nQ4, numNeighbors = mtxfac_sr.gd_kNN(newR, U4, V4, SG, 100, alpha, lamb, beta, L_C, list_index)
                            # #
                            # # time_exp4 = (time.time() - start_time) / 60
                            # #
                            # # exp4=mtxfac_sr.val(validation_index,newR, nP4, nQ4)
                            # #
                            # # f_result.write('SGD_kNN: ' + `exp4` + '\n')
                            # # f_result.write('time  : ' + `time_exp4` + '\n')
                            # # f_result.write('k :' + `numNeighbors` + '\n')
                            # #
                            # # ws.write(rowNumber, 9, exp4)
                            # # ws.write(rowNumber, 10, time_exp4)
                            # # # lenOfNeighbor = len(numNeighbors)
                            # # # for i in xrange(lenOfNeighbor):
                            # # #     ws.write(rowNumber, 11 + i, numNeighbors[i])
                            # #
                            # # print('k :' + `numNeighbors` + '\n')
                            # # print('SGD_kNN: ' + `exp4` + '\n')
                            # # print('time  : ' + `time_exp4` + '\n')
                            # # print '******************* FINISH SGD_kNN *******************'
                            #
                            print '******************* SGD_kNN BEGIN *******************'
                            L_C = 1
                            f_result.write('L_C : ' + `L_C` + '\n')
                            rowNumber = rowNumber + 1
                            ws.write(rowNumber, 0, rowNumber)
                            ws.write(rowNumber, 1, bound)
                            ws.write(rowNumber, 2, steps)
                            ws.write(rowNumber, 3, alpha)
                            ws.write(rowNumber, 4, lamb)
                            ws.write(rowNumber, 5, beta)
                            ws.write(rowNumber, 6, L_C)
                            weightVector, numNeighbors = mtxfac_sr.kNN(SG, L_C)
                            start_time = time.time()

                            nP5, nQ5, numNeighbors = mtxfac_sr.gd_kNN(R, U5, V5, SG, 50, alpha, lamb, beta, numNeighbors,
                                                                      list_index, weightVector)

                            time_exp5 = (time.time() - start_time) / 60

                            exp5=mtxfac_sr.val(validation_index,R, nP5, nQ5)

                            f_result.write('SGD_kNN: ' + `exp5` + '\n')
                            f_result.write('time  : ' + `time_exp5` + '\n')
                            f_result.write('k :' + `numNeighbors` + '\n')

                            ws.write(rowNumber, 9, exp5)
                            ws.write(rowNumber, 10, time_exp5)
                            # lenOfNeighbor = len(numNeighbors)
                            # for i in xrange(lenOfNeighbor):
                            #     ws.write(rowNumber, 11 + i, numNeighbors[i])

                            print('k :' + `numNeighbors` + '\n')
                            print('SGD_kNN: ' + `exp5` + '\n')
                            print('time  : ' + `time_exp5` + '\n')
                            print '******************* FINISH SGD_kNN *******************'
                            #
                            # # print '******************* SGD_kNN BEGIN *******************'
                            # # L_C = 0.1
                            # # f_result.write('L_C : ' + `L_C` + '\n')
                            # # rowNumber = rowNumber + 1
                            # # ws.write(rowNumber, 0, rowNumber)
                            # # ws.write(rowNumber, 1, bound)
                            # # ws.write(rowNumber, 2, steps)
                            # # ws.write(rowNumber, 3, alpha)
                            # # ws.write(rowNumber, 4, lamb)
                            # # ws.write(rowNumber, 5, beta)
                            # # ws.write(rowNumber, 6, L_C)
                            # # start_time = time.time()
                            # #
                            # # nP6, nQ6, numNeighbors = mtxfac_sr.gd_kNN(newR, U6, V6, SG, 100, alpha, lamb, beta, L_C, list_index)
                            # #
                            # # time_exp6 = (time.time() - start_time) / 60
                            # #
                            # # exp6=mtxfac_sr.val(validation_index,newR, nP6, nQ6)
                            # #
                            # # f_result.write('SGD_kNN: ' + `exp6` + '\n')
                            # # f_result.write('time  : ' + `time_exp6` + '\n')
                            # # f_result.write('k :' + `numNeighbors` + '\n')
                            # #
                            # # ws.write(rowNumber, 9, exp6)
                            # # ws.write(rowNumber, 10, time_exp6)
                            # # # lenOfNeighbor = len(numNeighbors)
                            # # # for i in xrange(lenOfNeighbor):
                            # # #     ws.write(rowNumber, 11 + i, numNeighbors[i])
                            # #
                            # # print('k :' + `numNeighbors` + '\n')
                            # # print('SGD_kNN: ' + `exp6` + '\n')
                            # # print('time  : ' + `time_exp6` + '\n')
                            # # print '******************* FINISH SGD_kNN *******************'
                            #
                            print '******************* SGD_kNN BEGIN *******************'
                            L_C = 5
                            f_result.write('L_C : ' + `L_C` + '\n')
                            rowNumber = rowNumber + 1
                            ws.write(rowNumber, 0, rowNumber)
                            ws.write(rowNumber, 1, bound)
                            ws.write(rowNumber, 2, steps)
                            ws.write(rowNumber, 3, alpha)
                            ws.write(rowNumber, 4, lamb)
                            ws.write(rowNumber, 5, beta)
                            ws.write(rowNumber, 6, L_C)
                            weightVector, numNeighbors = mtxfac_sr.kNN(SG, L_C)
                            start_time = time.time()

                            nP7, nQ7, numNeighbors = mtxfac_sr.gd_kNN(R, U7, V7, SG, 50, alpha, lamb, beta, numNeighbors,
                                                                      list_index, weightVector)

                            time_exp7 = (time.time() - start_time) / 60

                            exp7=mtxfac_sr.val(validation_index,R, nP7, nQ7)

                            f_result.write('SGD_kNN: ' + `exp7` + '\n')
                            f_result.write('time  : ' + `time_exp7` + '\n')
                            f_result.write('k :' + `numNeighbors` + '\n')

                            ws.write(rowNumber, 9, exp7)
                            ws.write(rowNumber, 10, time_exp7)
                            # lenOfNeighbor = len(numNeighbors)
                            # for i in xrange(lenOfNeighbor):
                            #     ws.write(rowNumber, 11 + i, numNeighbors[i])

                            print('k :' + `numNeighbors` + '\n')
                            print('SGD_kNN: ' + `exp7` + '\n')
                            print('time  : ' + `time_exp7` + '\n')
                            print '******************* FINISH SGD_kNN *******************'

                            # # print '******************* SGD_kNN BEGIN *******************'
                            # # L_C = 1
                            # # f_result.write('L_C : ' + `L_C` + '\n')
                            # # rowNumber = rowNumber + 1
                            # # ws.write(rowNumber, 0, rowNumber)
                            # # ws.write(rowNumber, 1, bound)
                            # # ws.write(rowNumber, 2, steps)
                            # # ws.write(rowNumber, 3, alpha)
                            # # ws.write(rowNumber, 4, lamb)
                            # # ws.write(rowNumber, 5, beta)
                            # # ws.write(rowNumber, 6, L_C)
                            # # start_time = time.time()
                            # #
                            # # nP8, nQ8, numNeighbors = mtxfac_sr.gd_kNN(newR, U8, V8, SG, 100, alpha, lamb, beta, L_C, list_index)
                            # #
                            # # time_exp8 = (time.time() - start_time) / 60
                            # #
                            # # exp8=mtxfac_sr.val(validation_index,newR, nP8, nQ8)
                            # #
                            # # f_result.write('SGD_kNN: ' + `exp8` + '\n')
                            # # f_result.write('time  : ' + `time_exp8` + '\n')
                            # # f_result.write('k :' + `numNeighbors` + '\n')
                            # #
                            # # ws.write(rowNumber, 9, exp8)
                            # # ws.write(rowNumber, 10, time_exp8)
                            # # # lenOfNeighbor = len(numNeighbors)
                            # # # for i in xrange(lenOfNeighbor):
                            # # #     ws.write(rowNumber, 11 + i, numNeighbors[i])
                            # #
                            # # print('k :' + `numNeighbors` + '\n')
                            # # print('SGD_kNN: ' + `exp8` + '\n')
                            # # print('time  : ' + `time_exp8` + '\n')
                            # # print '******************* FINISH SGD_kNN *******************'
                            #
                            print '******************* SGD_kNN BEGIN *******************'
                            L_C = 10
                            f_result.write('L_C : ' + `L_C` + '\n')
                            rowNumber = rowNumber + 1
                            ws.write(rowNumber, 0, rowNumber)
                            ws.write(rowNumber, 1, bound)
                            ws.write(rowNumber, 2, steps)
                            ws.write(rowNumber, 3, alpha)
                            ws.write(rowNumber, 4, lamb)
                            ws.write(rowNumber, 5, beta)
                            ws.write(rowNumber, 6, L_C)

                            weightVector, numNeighbors = mtxfac_sr.kNN(SG, L_C)
                            start_time = time.time()

                            nP9, nQ9, numNeighbors = mtxfac_sr.gd_kNN(R, U9, V9, SG, 50, alpha, lamb, beta, numNeighbors,
                                                                      list_index, weightVector)

                            time_exp9 = (time.time() - start_time) / 60

                            exp9=mtxfac_sr.val(validation_index,R, nP9, nQ9)

                            f_result.write('SGD_kNN: ' + `exp9` + '\n')
                            f_result.write('time  : ' + `time_exp9` + '\n')
                            f_result.write('k :' + `numNeighbors` + '\n')

                            ws.write(rowNumber, 9, exp9)
                            ws.write(rowNumber, 10, time_exp9)
                            # lenOfNeighbor = len(numNeighbors)
                            # for i in xrange(lenOfNeighbor):
                            #     ws.write(rowNumber, 11 + i, numNeighbors[i])

                            print('k :' + `numNeighbors` + '\n')
                            print('SGD_kNN: ' + `exp9` + '\n')
                            print('time  : ' + `time_exp9` + '\n')
                            print '******************* FINISH SGD_kNN *******************'
                            print '******************* SGD_kNN BEGIN *******************'
                            L_C = 50
                            f_result.write('L_C : ' + `L_C` + '\n')
                            rowNumber = rowNumber + 1
                            ws.write(rowNumber, 0, rowNumber)
                            ws.write(rowNumber, 1, bound)
                            ws.write(rowNumber, 2, steps)
                            ws.write(rowNumber, 3, alpha)
                            ws.write(rowNumber, 4, lamb)
                            ws.write(rowNumber, 5, beta)
                            ws.write(rowNumber, 6, L_C)

                            weightVector, numNeighbors = mtxfac_sr.kNN(SG, L_C)

                            start_time = time.time()

                            nP10, nQ10, numNeighbors = mtxfac_sr.gd_kNN(R, U10, V10, SG, 100, alpha, lamb, beta, numNeighbors,
                                                                      list_index, weightVector)

                            time_exp10 = (time.time() - start_time) / 60

                            exp10 = mtxfac_sr.val(validation_index, R, nP10, nQ10)

                            f_result.write('SGD_kNN: ' + `exp10` + '\n')
                            f_result.write('time  : ' + `time_exp10` + '\n')
                            f_result.write('k :' + `numNeighbors` + '\n')

                            ws.write(rowNumber, 9, exp10)
                            ws.write(rowNumber, 10, time_exp10)
                            # lenOfNeighbor = len(numNeighbors)
                            # for i in xrange(lenOfNeighbor):
                            #     ws.write(rowNumber, 11 + i, numNeighbors[i])

                            print('k :' + `numNeighbors` + '\n')
                            print('SGD_kNN: ' + `exp10` + '\n')
                            print('time  : ' + `time_exp10` + '\n')
                            print '******************* FINISH SGD_kNN *******************'
                            f_result.close()

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
    U = random.uniform(0,0.01,size=(3000, 12000))
    V = random.uniform(0,0.01,size=(20000, 12000))
    SN_FILE = '../dataset/user_friends'

    exp('NY', R, U, V, SN_FILE, False)

    # R = numpy.loadtxt(open("../dataset/NY_MATRIX","rb"),delimiter=",")
    # R = numpy.array(R)
    # U = numpy.loadtxt(open("../dataset/NY_U","rb"),delimiter=",")
    # V = numpy.loadtxt(open("../dataset/NY_V","rb"),delimiter=",")
    # SN_FILE = '../dataset/NY_SN'
    #
    # exp('NY', R, U, V, SN_FILE, True)
    #
    # R = numpy.loadtxt(open("../dataset/IL_MATRIX","rb"),delimiter=",")
    # R = numpy.array(R)
    # U = numpy.loadtxt(open("../dataset/IL_U","rb"),delimiter=",")
    # V = numpy.loadtxt(open("../dataset/IL_V","rb"),delimiter=",")
    # SN_FILE = '../dataset/IL_SN'
    #
    # exp('IL', R, U, V, SN_FILE, True)
    #
    # R = numpy.loadtxt(open("../dataset/CA_MATRIX","rb"),delimiter=",")
    # R = numpy.array(R)
    # U = numpy.loadtxt(open("../dataset/CA_U","rb"),delimiter=",")
    # V = numpy.loadtxt(open("../dataset/CA_V","rb"),delimiter=",")
    # SN_FILE = '../dataset/CA_SN'
    #
    # exp('CA', R, U, V, SN_FILE, True)
