import numpy
from random import randint
import time
from lib import stratus
from lib import util

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

def load_matrix_index_for_anotherDataSet(R, bound):
    size = 2100
    list_index = []
    validation_index = []
    newR = numpy.zeros((size, 20000))
    for i in xrange(len(R)):
        x = R[i][0]
        y = R[i][1]
        value = R[i][2]/1000
        if x<size:
            newR[x][y] = value
            randNumber = numpy.random.randint(0, 100)
            if randNumber <= bound:
                list_index.append(`x` + ',' + `y`)
            else:
                validation_index.append(`x` + ',' + `y`)

    return list_index, validation_index, newR

def gd_update(Rij, Ui, Vj, alpha, lamb):

    e = numpy.dot(Ui.T,Vj) - Rij

    u_temp = Ui - alpha * ( (e * Vj) + (lamb * Ui) )
    v_temp = Vj - alpha * ( (e * Ui) + (lamb * Vj) )


    return u_temp, v_temp

def gd(R, U, V, steps, alpha, lamb, list_index):


    for step in xrange(steps):

        for index in xrange(len(list_index)):

            sI,sJ =  list_index[index].split(',')

            i = int(float(sI))
            j = int(float(sJ))

            e = numpy.dot(U[i].T,V[j]) - R[i,j]

            u_temp = U[i] - alpha * ( (e * V[j]) + (lamb * U[i]) )
            V[j] = V[j] - alpha * ( (e * U[i]) + (lamb * V[j]) )
            U[i] = u_temp

    return U, V

def sgd(R, U, V, steps=1800000, alpha=0.0001, lamb=0.002):

    percent = 0
    current_percent = 0

    list_index = load_matrix_index(R)

    len_list_index = len(list_index)

    for step in xrange(steps):

        index = randint(0,len_list_index-1)

        sI,sJ =  list_index[index].split(',')

        i = int(sI)
        j = int(sJ)

        U[i], V[j], cost = gd_update(R[i][j], U[i,:], V[j,:], alpha, lamb)

        current_percent = util.calc_progress(steps, step+1, current_percent)

        if(current_percent != percent):
            print current_percent
            percent = current_percent

    return U, V


def dsgd(R, U, V, stratus_number, T, steps, alpha, lamb):

    percent = 0
    current_percent = 0

    for step in xrange(T):

        list_stratus, list_U, list_V, index_pointer_c = stratus.split_matrix(R, U, V, stratus_number, step)

        for i in xrange(stratus_number):

            list_U[i],list_V[i] = sgd(list_stratus[i], list_U[i], list_V[i], steps, alpha, lamb)

        index_U=0
        for index_array in xrange(stratus_number):
            temp_U = list_U[index_array]
            
            for i in xrange(len(temp_U)):
                for j in xrange(len(temp_U[0])):
                    U[index_U][j] = temp_U[i][j]
                index_U += 1

        index_V=0
        for x in xrange(stratus_number):
            index_V = index_pointer_c[x]

            temp_V = list_V[x]

            for i in xrange(len(temp_V)):
                V[index_V+i] = temp_V[i]

        current_percent = util.calc_progress(T, step+1, current_percent)

        if(current_percent != percent):
            print current_percent
            percent = current_percent

    return U, V

def dgd(R, U, V, stratus_number, T, steps, alpha, lamb):

    percent = 0
    current_percent = 0

    for step in xrange(T):

        list_stratus, list_U, list_V, index_pointer_r, index_pointer_c = stratus.split_matrix(R, U, V, stratus_number, step)

        for i in xrange(stratus_number):

            list_U[i],list_V[i], cost_f = gd(list_stratus[i], list_U[i], list_V[i], steps, alpha, lamb)

        index_U=0
        for index_array in xrange(stratus_number):
            temp_U = list_U[index_array]
            
            for i in xrange(len(temp_U)):
                for j in xrange(len(temp_U[0])):
                    U[index_U][j] = temp_U[i][j]
                index_U += 1

        index_V=0
        for x in xrange(stratus_number):
            index_V = index_pointer_c[x]

            temp_V = list_V[x]

            for i in xrange(len(temp_V)):
                V[index_V+i] = temp_V[i]

        current_percent = util.calc_progress(T, step+1, current_percent)

        if(current_percent != percent):
            print current_percent
            percent = current_percent

    return U, V