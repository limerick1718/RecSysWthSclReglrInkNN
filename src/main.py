from lib import kNN_Utils
from lib import OlderAlgo
from lib import Utilities
from lib import FriendsRegular
import numpy
import time
import xlwt

# import pylab
import datetime
from numpy import random

global  date
date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
global f_result
global rowNumber
rowNumber = 0
wb = xlwt.Workbook()
global ws
ws = wb.add_sheet('RssrkNN')
global userNumber
global itemNumber

def preWrite(rowNumber, ws,f_result, bound, alpha, lamb, beta, L_C):
    f_result.write('L_C : ' + `L_C` + '\n')
    ws.write(rowNumber, 0, rowNumber)
    ws.write(rowNumber, 1, bound)
    ws.write(rowNumber, 3, alpha)
    ws.write(rowNumber, 4, lamb)
    ws.write(rowNumber, 5, beta)
    ws.write(rowNumber, 6, L_C)

def afterWrite(ws,f_result, validation_index, newR, time_exp, numNeighbors, steps):
    start_timeR = time.time()
    rmse = Utilities.rmse(validation_index, newR, U, V)
    time_R = (time.time() - start_timeR) / 60
    start_timeM = time.time()
    mae = Utilities.mae(validation_index, newR, U, V)
    time_M = (time.time() - start_timeM) / 60
    f_result.write('rmse: ' + `rmse` + '\n')
    f_result.write('mae: ' + `mae` + '\n')
    f_result.write('time  : ' + `time_exp` + '\n')
    f_result.write('k :' + `numNeighbors` + '\n')
    f_result.write('time_R  : ' + `time_R` + '\n')
    f_result.write('time_M  : ' + `time_M` + '\n')

    ws.write(rowNumber, 9, rmse)
    ws.write(rowNumber, 10, time_exp)
    ws.write(rowNumber, 2, steps)
    ws.write(rowNumber, 11, mae)
    ws.write(rowNumber, 12, time_R)
    ws.write(rowNumber, 13, time_M)

    print('k :' + `numNeighbors` + '\n')
    print('rmse: ' + `rmse` + '\n')
    print('mae: ' + `mae` + '\n')
    print('time  : ' + `time_exp` + '\n')
    print('time_R  : ' + `time_R` + '\n')
    print('time_M  : ' + `time_M` + '\n')
    print ('steps :' + `steps` + '\n')

def rskNN(bound, alpha, lamb, beta, U, V, L_C, list_index, validation_index, newR, SG):
    global rowNumber
    global ws
    print '******************* SGD_kNN BEGIN *******************'

    rowNumber = rowNumber + 1
    preWrite(rowNumber, ws,f_result, bound, alpha, lamb, beta, L_C)
    start_time = time.time()

    U, V, numNeighbors, steps = kNN_Utils.RskNN(list_index, alpha, 0.001, 0.001, 0.001, 0.1,U, V, L_C, newR, SG)

    time_exp = (time.time() - start_time) / 60
    afterWrite(ws,f_result, validation_index, newR, time_exp, numNeighbors, steps)
    print '******************* FINISH SGD_kNN *******************'

def gdOld(bound, alpha, lamb, beta, U, V, list_index, validation_index, newR, SG):
    global rowNumber
    global ws
    print '******************* SGD_kNN BEGIN *******************'

    rowNumber = rowNumber + 1
    preWrite(rowNumber, ws,f_result, bound, alpha, lamb, beta, 0)

    start_time = time.time()

    U, V, steps = OlderAlgo.gd_default(newR, U, V, SG, alpha, lamb, beta, list_index)
    time_exp = (time.time() - start_time) / 60
    afterWrite(ws,f_result, validation_index, newR, time_exp, 0, steps)

    print '******************* FINISH SGD_kNN *******************'

def FR(bound, alpha, lamb, beta, U, V, list_index, validation_index, newR, SG):
    global rowNumber
    global ws
    print '******************* SGD_kNN BEGIN *******************'

    rowNumber = rowNumber + 1
    preWrite(rowNumber, ws,f_result, bound, alpha, lamb, beta, 0)

    start_time = time.time()

    U, V, steps = FriendsRegular.FR(list_index, alpha, 0.001, 0.001, 0.001, 0.1, U, V, newR, SG)
    time_exp = (time.time() - start_time) / 60
    afterWrite(ws,f_result, validation_index, newR, time_exp, 0, steps)

    print '******************* FINISH SGD_kNN *******************'

def loadFile(isTraditionalFile, bound, R, social_network):
    global userNumber
    global itemNumber
    if isTraditionalFile:
        list_index, validation_index = Utilities.load_matrix_index(R, bound)
    else:
        list_index, validation_index, newR = Utilities.load_matrix_index_for_anotherDataSet(R, bound, userNumber, itemNumber)

    if isTraditionalFile:
        SG = Utilities.load_grafo_social(R, social_network)
    else:
        SG, SGS = Utilities.load_grafo_social_for_anotherDataSet(newR, social_network, userNumber, itemNumber)
    return list_index, validation_index, newR, SG, SGS

def excelHelper(ws):
    ws.write(0, 0, 'rowNumber')
    ws.write(0, 1, 'bound')
    ws.write(0, 2, 'step')
    ws.write(0, 3, 'alpha')
    ws.write(0, 4, 'lamb')
    ws.write(0, 5, 'beta')
    ws.write(0, 6, 'L_C')
    ws.write(0, 7, 'SGD_LOSS')
    ws.write(0, 8, 'SGD_time')
    ws.write(0, 9, 'RMSE')
    ws.write(0, 10, 'SGD_kNN_time')
    ws.write(0, 11, 'MAE')
    ws.write(0, 12, 'timeR')
    ws.write(0, 13, 'timeM')


def exp(exp_name, R, U, V, SN_FILE, isTraditionalFile):
    global ws
    global rowNumber
    global f_result
    global userNumber
    global itemNumber

    userNumber = len(U)
    itemNumber = len(V)

    Bound = [40,60,80]
    Alpha = [0.001]
    Lamb  = [0.01]
    Beta  = [0.001]
    L_CRatio = [0.001,0.01,0.1,1,10,100,1000,10000]

    excelHelper(ws)

    for bound in Bound:
        list_index, validation_index, newR, SG, SGS = loadFile(isTraditionalFile, bound, R, SN_FILE)
        for alpha in Alpha:
            for lamb in Lamb:
                for beta in Beta:
                    gdOld(bound, alpha, lamb, beta, U, V, list_index, validation_index, newR, SGS)
                    FR(bound, alpha, lamb, beta, U, V, list_index, validation_index, newR, SG)
                    for L_C in L_CRatio:
                        rskNN(bound, alpha, lamb, beta, U, V, L_C, list_index, validation_index, newR, SG)
                    print '******************* FINISH L_C *******************'
                print '******************* FINISH Beta *******************'
            print '******************* FINISH Lamb *******************'
        print '******************* FINISH Alpha *******************'
    f_result.close()
    wb.save('../resultSet/lastfm' + date+'.xls')
    print '******************* FINISH *******************'



if __name__ == "__main__":

    '************************ EXP GD x GDRS ***************************'
    # global f_result
    # f_result = open('../resultSet/lastfm' + '_' + date, 'w')
    # R = numpy.loadtxt(open("../dataset/user_artists", "rb"), delimiter='\t')
    # R = numpy.array(R)
    # U = random.uniform(0,0.01,size=(2100, 100))
    # V = random.uniform(0,0.01,size=(20000, 100))
    # SN_FILE = '../dataset/user_friends'
    #
    # exp('lastfm', R, U, V, SN_FILE, False)

    global f_result
    f_result = open('../resultSet/Epinions' + '_' + date, 'w')
    R = numpy.loadtxt(open("../dataset/epinions/ratings_data.txt", "rb"), delimiter=' ')
    R = numpy.array(R)
    U = random.uniform(0,0.01,size=(50000, 100))
    V = random.uniform(0,0.01,size=(200000, 100))
    SN_FILE = '../dataset/epinions/trust_data.txt'
    social_network = numpy.loadtxt(open(SN_FILE, "rb"), delimiter=' ')

    exp('lastfm', R, U, V, social_network, False)
    #
    # R = numpy.loadtxt(open("../dataset/hetrec2011-delicious-2k/user_taggedbookmarks", "rb"), delimiter='\t')
    # R = numpy.array(R)
    # U = random.uniform(0,0.01,size=(1000, 100))
    # V = random.uniform(0,0.01,size=(70000, 100))
    # SN_FILE = '../dataset/hetrec2011-delicious-2k/user_contacts'
    #
    # exp('delicious', R, U, V, SN_FILE, False)

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