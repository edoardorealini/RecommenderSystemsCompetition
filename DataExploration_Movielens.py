URM_file = open("data/Movielens_10M/ml-10M100K/ratings.dat", 'r')

print(type(URM_file))

for _ in range(10):
    print(URM_file.readline())

URM_file.seek(0)
numberInteractions = 0

for _ in URM_file:
    numberInteractions += 1

print("The number of interactions is {}".format(numberInteractions))

#splitting the data

def rowSplit(rowString):
    split = rowString.split("::")
    split[3] = split[3].replace("\n", "")

    split[0] = int(split[0])    #UserID
    split[1] = int(split[1])    #MovieID
    split[2] = float(split[2])  #Rating
    split[3] = int(split[3])    #Timestamp

    result = tuple(split)
    return result

#mi metto all'inizio del file
URM_file.seek(0)
#dichiaro la lista che contiene le righe come tuple
URM_tuples = []

for line in URM_file:
    URM_tuples.append(rowSplit(line))

print(URM_tuples[0:10])

#separiamo le colonne in 3 liste indipendenti
userList, itemList, ratingList, timestampList = zip(*URM_tuples)

userList = list(userList)
itemList = list(itemList)
ratingList = list(ratingList)
timestampList = list(timestampList)

print(userList[0:10])
print(itemList[0:10])
print(ratingList[0:10])
print(timestampList[0:10])

userList_unique = list(set(userList))
itemList_unique = list(set(itemList))

numUsers = len(userList_unique)
numItems = len(itemList_unique)


print ("Number of items\t {}, Number of users\t {}".format(numItems, numUsers))
print ("Max ID items\t {}, Max Id users\t {}\n".format(max(itemList_unique), max(userList_unique)))
print ("Average interactions per user {:.2f}".format(numberInteractions/numUsers))
print ("Average interactions per item {:.2f}\n".format(numberInteractions/numItems))

print ("Sparsity {:.2f} %".format((1-float(numberInteractions)/(numItems*numUsers))*100))

import matplotlib.pyplot as pyplot

# Clone the list to avoid changing the ordering of the original data
timestamp_sorted = list(timestampList)
timestamp_sorted.sort()


pyplot.plot(timestamp_sorted, 'ro')
pyplot.ylabel('Timestamp ')
pyplot.xlabel('Item Index')
pyplot.show()

#creating the data structures

import scipy.sparse as sps

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
print(type(URM_all))

#converting to CSR
URM_all.tocsr()