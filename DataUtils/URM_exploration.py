import scipy.sparse as sps

URM_file = open("data/competition/data_train.csv", 'r')

print(type(URM_file))

for _ in range(10):
    print(URM_file.readline())

URM_file.seek(0)
numberInteractions = 0

for _ in URM_file:
    numberInteractions += 1

print("The number of interactions is {}".format(numberInteractions))

def rowSplit(rowString):
    split = rowString.split(",")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = float(split[2])

    result = tuple(split)
    return result

URM_file.seek(0)
URM_tuples = []

for line in URM_file:
    URM_tuples.append(rowSplit(line))

print(URM_tuples[0:10])

userList, itemList, ratingList = zip(*URM_tuples)
userList = list(userList)
itemList = list(itemList)
ratingList = list(ratingList)

print("Users:\t\t", userList[0:10])
print("Items:\t\t", itemList[0:10])
print("Ratings:\t", ratingList[0:10])

userList_unique = list(set(userList))
itemList_unique = list(set(itemList))

numUsers = len(userList_unique)
numItems = len(itemList_unique)


print ("Number of items\t {}, Number of users\t {}".format(numItems, numUsers))
print ("Max ID items\t {}, Max Id users\t {}\n".format(max(itemList_unique), max(userList_unique)))
print ("Average interactions per user {:.2f}".format(numberInteractions/numUsers))
print ("Average interactions per item {:.2f}\n".format(numberInteractions/numItems))

print ("Sparsity {:.2f} %".format((1-float(numberInteractions)/(numItems*numUsers))*100))

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
print(type(URM_all))

#converting to CSR
URM_all.tocsr()
print(type(URM_all))
print("Correctly imported data and converted into csr type!")

sps.save_npz('data/competition/sparse_URM.npz', URM_all, compressed=True)
print("Matrix saved in sparse_URM.npz")