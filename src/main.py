from lib import kNN_Utils
from lib import OlderAlgo
from lib import Utilities
from lib import FriendsRegular
from sklearn.model_selection import KFold
from scipy.sparse import linalg as la

import numpy
import time
import xlwt
import dataProcessor
import datetime

global expName
global  date
date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
global rowNumber
rowNumber = 0
wb = xlwt.Workbook()
global ws
ws = wb.add_sheet('RssrkNN')
global userNumber
global itemNumber


def fileWriter(values):
    global date
    global expName
    f_result = open('../resultSet/NY_SN'+expName+date, 'a')
    f_result.write( values + "\n")
    f_result.close()

def preWrite(rowNumber, ws,alpha, lamb, beta, L_C):
    fileWriter('L_C : ' + `L_C` + '\n')
    ws.write(rowNumber, 0, rowNumber)
    # ws.write(rowNumber, 1, bound)
    ws.write(rowNumber, 3, alpha)
    ws.write(rowNumber, 4, lamb)
    ws.write(rowNumber, 5, beta)
    ws.write(rowNumber, 6, L_C)

def afterWrite(ws,R_test, time_exp, numNeighbors, steps, U, V):
    global rowNumber
    start_timeR = time.time()
    rmse = Utilities.rmse(R_test, U, V)
    time_R = (time.time() - start_timeR) / 60
    start_timeM = time.time()
    mae = Utilities.mae(R_test, U, V)
    time_M = (time.time() - start_timeM) / 60
    fileWriter('rmse: ' + `rmse` + '\n')
    fileWriter('mae: ' + `mae` + '\n')
    fileWriter('time  : ' + `time_exp` + '\n')
    fileWriter('k :' + `numNeighbors` + '\n')
    fileWriter('time_R  : ' + `time_R` + '\n')
    fileWriter('time_M  : ' + `time_M` + '\n')

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

def rskNN(alpha, lamb, beta, U, V, L_C, RTrain, RTest, SG):
    global rowNumber
    global ws
    global userNumber
    print '******************* SGD_kNN BEGIN *******************'
    R_train = RTrain
    R_test = RTest
    U_rs = U
    V_rs = V
    rowNumber = rowNumber + 1
    preWrite(rowNumber, ws,alpha, lamb, beta, L_C)
    start_time = time.time()

    U_KNN, V_KNN, numNeighbors, steps = kNN_Utils.RskNN(R_train, alpha, 0.001, 0.001, 0.001, 0.1,U_rs, V_rs, L_C, SG, userNumber)

    time_exp = (time.time() - start_time) / 60
    afterWrite(ws,R_test, time_exp, numNeighbors, steps,U_KNN,V_KNN)
    print '******************* FINISH SGD_kNN *******************'

def social_regularization(alpha, lamb, beta, U, V, RTrain, RTest, SG):
    global rowNumber
    global ws
    print '******************* SGD_kNN BEGIN *******************'
    R_train = RTrain
    R_test = RTest

    U_tempSR = U
    V_tempSR = V
    rowNumber = rowNumber + 1
    preWrite(rowNumber, ws,alpha, lamb, beta, 0)

    start_time = time.time()

    U_gdOld, V_gdOld, steps = OlderAlgo.social_regular(R_train, U_tempSR, V_tempSR, SG, alpha, lamb, beta)
    time_exp = (time.time() - start_time) / 60
    afterWrite(ws, R_test, time_exp, 0, steps, U_gdOld, V_gdOld)

    print '******************* FINISH SGD_kNN *******************'

def FR(alpha, lamb, beta, U, V, RTrain, RTest, SG):
    global rowNumber
    global ws
    global userNumber
    print '******************* SGD_kNN BEGIN *******************'

    R_train = RTrain
    R_test = RTest
    U_FR = U
    V_FR = V
    rowNumber = rowNumber + 1
    preWrite(rowNumber, ws,alpha, lamb, beta, 0)

    start_time = time.time()

    U_FR, V_FR, steps = FriendsRegular.FR(R_train, alpha, 0.001, 0.001, 0.001, 0.1, U_FR, V_FR, SG, userNumber)
    time_exp = (time.time() - start_time) / 60
    afterWrite(ws, R_test, time_exp, 0, steps,U_FR, V_FR)

    print '******************* FINISH SGD_kNN *******************'

def loadFile(R, social_network):
    print "*****************begin load File***********************"
    global userNumber
    global itemNumber
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(R):
        R_trian, R_test = R[train_index], R[test_index]
    SGS = Utilities.load_grafo_social_for_anotherDataSet(R, social_network, userNumber)
    return R_trian, R_test,SGS

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


def svd(R_train, R_test):
    global rowNumber
    rowNumber = rowNumber + 1
    print "*************begin svd*******************"
    U_SVD, sigma, V_SVD = la.svds(R_train, 10)
    afterWrite(ws, R_test, 0, 0, 0, U_SVD, V_SVD)

def exp(ExpName,R, U, V, SN_FILE):
    print "********************exp begin********************************"
    global ws
    global rowNumber
    global userNumber
    global itemNumber
    global expName
    expName = ExpName
    userNumber = len(U) + 1
    itemNumber = len(V) + 1

    Alpha = [0.001]
    Lamb  = [0.01]
    Beta  = [0.001]
    L_CRatio = [0.001,0.01,0.1,1,10,100,1000,10000]

    excelHelper(ws)

    RTrain, RTest, SGS= loadFile(R, SN_FILE)
    R_train = RTrain
    R_test = RTest
    for alpha in Alpha:
        for lamb in Lamb:
            for beta in Beta:
                R_train1 = RTrain
                R_test1 = RTest
                svd( R_train1, R_test1)
                social_regularization(alpha, lamb, beta, U, V, R_train, R_test, SGS)
                FR(alpha, lamb, beta, U, V, R_train, R_test, SGS)
                for L_C in L_CRatio:
                    rskNN(alpha, lamb, beta, U, V, L_C, R_train, R_test, SGS)
                print '******************* FINISH L_C *******************'
            print '******************* FINISH Beta *******************'
        print '******************* FINISH Lamb *******************'
    print '******************* FINISH Alpha *******************'
    wb.save('../resultSet/lastfm' + date+'.xls')
    print '******************* FINISH *******************'



if __name__ == "__main__":

    '************************ EXP GD x GDRS ***************************'
    # # global f_result
    # # f_result = open('../resultSet/lastfm' + '_' + date, 'w')
    # # R = numpy.loadtxt(open("../dataset/user_artists", "rb"), delimiter='\t')
    # # R = numpy.array(R)
    # R, U, V = dataProcessor.dataStatic("../dataset/user_artists",'\t')
    # # U = random.uniform(0,0.01,size=(2200, 100))
    # # V = random.uniform(0,0.01,size=(20000, 100))
    # SN_FILE = '../dataset/user_friends'
    # social_network = numpy.loadtxt(open(SN_FILE, "rb"), delimiter='\t')
    # #
    # exp('lastfm', R, U, V, social_network)
    #
    # R, U, V = dataProcessor.dataStatic("../dataset/epinions/ratings_data",' ')
    # SN_FILE = '../dataset/epinions/trust_data'
    # social_network = numpy.loadtxt(open(SN_FILE, "rb"), delimiter=' ')
    # exp('epinions', R, U, V, social_network)

    # R, U, V = dataProcessor.dataStatic("../dataset/hetrec2011-lastfm-2k/user_artists", '\t')
    # SN_FILE = '../dataset/hetrec2011-lastfm-2k/user_friends'
    # social_network = numpy.loadtxt(open(SN_FILE, "rb"), delimiter='\t')
    # exp('epinions', R, U, V, social_network)

    # global f_result
    # f_result = open('../resultSet/lastfm' + '_' + date, 'w')
    # R = numpy.loadtxt(open("../dataset/user_artists", "rb"), delimiter='\t')
    # R = numpy.array(R)
    # U = random.uniform(0,0.01,size=(2200, 100))
    # V = random.uniform(0,0.01,size=(20000, 100))
    # SN_FILE = '../dataset/user_friends'
    # social_network = numpy.loadtxt(open(SN_FILE, "rb"), delimiter='\t')
    #
    # exp('lastfm', R, U, V, social_network, False)

    # global f_result
    # f_result = open('../resultSet/Epinions' + '_' + date, 'w')
    # R = numpy.loadtxt(open("../dataset/epinions/ratings_data.txt", "rb"), delimiter=' ')
    # R = numpy.array(R)
    # U = random.uniform(0,0.01,size=(50000, 100))
    # V = random.uniform(0,0.01,size=(200000, 100))
    # SN_FILE = '../dataset/epinions/trust_data.txt'
    # social_network = numpy.loadtxt(open(SN_FILE, "rb"), delimiter=' ')
    #
    # exp('lastfm', R, U, V, social_network, False)
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
    R, U, V = dataProcessor.dataStatic("../dataset/NY_MATRIX", ',',False)
    U = numpy.loadtxt(open("../dataset/NY_U","rb"),delimiter=",")
    V = numpy.loadtxt(open("../dataset/NY_V","rb"),delimiter=",").T
    SN_FILE = '../dataset/NY_SN'
    social_network = numpy.loadtxt(open(SN_FILE, "rb"), delimiter=',')
    exp('NY_MATRIX', R, U, V, social_network)
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