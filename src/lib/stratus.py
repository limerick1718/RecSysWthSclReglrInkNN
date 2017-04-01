import numpy
from random import randint

def split_matrix(R, U, V, block_size, step):
   
    R = numpy.array(R)

    M = len(R)
    N = len(R[0])

    index_r = []
    index_c = []

    r = M/block_size
    c = N/block_size

    if M%block_size == 0:
        while len(index_r) < block_size:
            index_r.append(M/block_size)
    else:
        r = int(r)
        while len(index_r) < block_size:
            if len(index_r) + 1 == block_size:
                index_r.append(M - r*(block_size - 1))
            else:
                index_r.append(r)

    if N%block_size == 0:
        while len(index_c) < block_size:
            index_c.append(N/block_size)
    else:
        c = int(c)
        while len(index_c) < block_size:
            if len(index_c) + 1 == block_size:
                index_c.append(N - c*(block_size - 1))
            else:
                index_c.append(c)

    index_stratus = numpy.zeros( (block_size,block_size) )

    pointer_r = 0
    pointer_c = 0

    for i in xrange(block_size):
        for j in xrange(block_size):
            index_stratus[i][j] = float(`pointer_r`+'.'+`pointer_c`)
            pointer_c += index_c[j]
        pointer_c = 0
        pointer_r += index_r[i]

    index_stratus_selected = []
    index_stratus_selected.append(randint(0,block_size-1))
    while len(index_stratus_selected) < block_size:
        index = randint(0,block_size-1)
        valid = True
        for i in xrange(len(index_stratus_selected)):
            if index_stratus_selected[i] == index:
                valid = False
                break
        if valid:
            index_stratus_selected.append(index)

    list_stratus = []

    splited_U = []
    splited_V = []

    index_pointer_r = []
    index_pointer_c = []

    for k in xrange(block_size):
        
        pointer_r, pointer_c = str(index_stratus[k][index_stratus_selected[k]]).split('.')

        pointer_r = int(pointer_r)
        pointer_c = int(pointer_c)

        index_pointer_r.append(pointer_r)
        index_pointer_c.append(pointer_c)
        
        if(M>=N):
            total_r = index_r[index_stratus_selected[k]]
            total_c = index_c[index_stratus_selected[k]]
        else:
            total_r = index_r[k]
            total_c = index_c[k]

        stratus = numpy.zeros( (total_r,total_c) )

        temp_splited_U = numpy.zeros( (total_r,len(U[0])) )
        temp_splited_V = numpy.zeros( (total_c,len(V[0])) )

        loop_c = 0

        for i in xrange(total_r):

            for j in xrange(total_c):
                stratus[i][j] = R[pointer_r][pointer_c]

                if loop_c < total_c:
        			for x in xrange(len(V[0])):
        				temp_splited_V[j][x] = V[pointer_c][x]

				loop_c += 1

                pointer_c += 1

            for j in xrange(len(U[0])):
            	temp_splited_U[i][j] = U[pointer_r][j]

            pointer_c = pointer_c - total_c
            pointer_r += 1

        splited_U.append(temp_splited_U)
        splited_V.append(temp_splited_V)

        list_stratus.append(stratus)

    return list_stratus, splited_U, splited_V, index_pointer_r, index_pointer_c