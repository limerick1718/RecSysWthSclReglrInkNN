import numpy

from lib import util

# R = [
#     	[5,4,0,1,3],
#         [5,0,0,0,0],
#         [0,5,0,1,1],
#         [1,0,0,5,4],
#         [0,1,0,0,5],
#         [1,0,0,5,0]
#     ]

R = numpy.loadtxt(open("/home/arthur/projects/mestrado/bigdata/foursquare/IL_MATRIX","rb"),delimiter=",")

print 'Number of users: '+`len(R)`
print 'Number of items: '+`len(R[0])`

'******************** User Statistic *********************'

count_min = 1000000
count_max = 0
sum_user_rating = 0

for i in xrange(len(R)):

	count = 0

	for j in xrange(len(R[0])):
		if R[i][j] > 0:
			count += 1

	sum_user_rating += count

	if count < count_min:
		count_min = count

	if count > count_max:
		count_max = count

print 'Min user rating: '+`count_min`
print 'Max user rating: '+`count_max`
print 'AVG user rating: '+`sum_user_rating / len(R)`

'******************** Item Statistic *********************'

count_min = 1000000
count_max = 0
sum_item_rating = 0

for i in xrange(len(R[0])):

	count = 0

	for j in xrange(len(R)):
		if R[j][i] > 0:
			count += 1

	sum_item_rating += count

	if count < count_min:
		count_min = count

	if count > count_max:
		count_max = count

print 'Min item rating: '+`count_min`
print 'Max item rating: '+`count_max`
print 'AVG item rating: '+`sum_item_rating / len(R[0])`

'******************** SN Statistic *********************'

sum_cor_pearson = 0

grafo_size = len(R)

SG = numpy.zeros((grafo_size, grafo_size))

social_network = numpy.loadtxt(open("/home/arthur/projects/mestrado/bigdata/foursquare/IL_SN","rb"),delimiter=",")

for i in xrange(len(social_network)):
    
    user   = social_network[i][0]
    friend = social_network[i][1]

    x = R[user]
    y = R[friend]

    cor_pearson = util.pearson(x,y)

    if cor_pearson > 0:
    	sum_cor_pearson += 1

    SG[user][friend] = 1
    SG[friend][user] = 1

count_min = 1000000
count_max = 0
sum_friends = 0

for i in xrange(len(SG)):

	count = 0

	for j in xrange(len(SG[0])):
		if SG[i][j] > 0:
			count += 1

	sum_friends += count

	if count < count_min:
		count_min = count

	if count > count_max:
		count_max = count

print 'Min friends: '+`count_min`
print 'Max friends: '+`count_max`
print 'AVG friends: '+`sum_friends / len(SG)`
print 'Total SIM: '+`sum_cor_pearson`
	

