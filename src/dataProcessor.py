import numpy
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as la

def dataStatic(path, Symbol):
    R = numpy.array(numpy.loadtxt(open(path, "rb"), delimiter=Symbol))
    userIndex = []
    itemIndex = []
    rankingScores = []
    maxUser = 0
    maxItem = 0
    maxScore = 0
    for i in xrange(len(R)):
        userIndex.append(int(float(R[i][0])))
        if R[i][0] > maxUser:
            maxUser = R[i][0]
        itemIndex.append(int(float(R[i][1])))
        if R[i][1] > maxItem:
            maxItem = R[i][1]
        rankingScores.append(float(R[i][2]))
        if R[i][2] > maxScore:
            maxScore = R[i][2]
    rankingScores = [k * 5.0/ maxScore * 1.0 for k in rankingScores]
    userList = numpy.array(userIndex)
    itemList = numpy.array(itemIndex)
    rankingList = numpy.array(rankingScores)
    newR = csr_matrix((rankingList,(userList,itemList)),shape=(maxUser + 1, maxItem + 1))
    print "maxUser"
    print maxUser
    print "maxItem"
    print maxItem

    # demesion = userNumber + 1 \times itemNumber + 1
    # because 0 row 0 column is space
    U,sigma,V = la.svds(newR, 10)
    # U, V is +1 dimension
    # fileU = open('../dataset/hetrec2011-lastfm-2k/U', 'w')
    # fileU.write(U)
    # # print U
    # fileV = open('../dataset/hetrec2011-lastfm-2k/V', 'w')
    # fileV.write(V)
    # # print V
    # fileR = open('../dataset/hetrec2011-lastfm-2k/R', 'w')
    # fileR.write(newR.toarray())
    # print R
    return newR, U, V



