from DataUtils import datasetSplitter
import scipy.sparse as sps
import time


splitter = datasetSplitter()
splitter.loadURMdata("data/competition/data_train.csv")

splitter.userList_toString(to=30)
splitter.itemList_toString(to=30)
splitter.userListUnique_toString(to=30)
splitter.itemListUnique_toString(to=30)

start_time = time.time()
splitter.splitData()
end_time = time.time()
print("Split time: {:.2f} min".format((end_time-start_time)/60))

splitter.userListUnique_toString(to=30)
splitter.itemListUnique_toString(to=30)
splitter.userList_toString(to=30)
splitter.itemList_toString(to=30)
splitter.userListTest_toString(to=30)
splitter.itemListTest_toString(to=30)

URM_test = sps.load_npz("data/competition/URM_test.npz")
URM_test.tocsr()

URM_train = sps.load_npz("data/competition/URM_train.npz")
URM_train.tocsr()

array_test = URM_test.toarray()
array_train = URM_train.toarray()

print("URM_test: " , array_test)
print("URM_train: ", array_train)