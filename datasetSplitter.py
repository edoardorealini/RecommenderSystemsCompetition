# With this class we want to split the URM into train, validation and test sets
# The main idea is to split the dataset as follows:
#   - The test set is created starting from the full URM. For each user is collected one interaction and stored in the
#       test set.
import scipy.sparse as sps
import random
from ParserURM import ParserURM


class datasetSplitter():

    # Splittin percentage not needed
    # Taking a random interaction for each user
    def __init__(self):
        self.userList = []
        self.itemList = []
        self.userList_unique = []
        self.itemList_unique = []
        self.userList_testSet = []
        self.itemList_testSet = []
        self.userList_trainSet = []
        self.itemList_trainSet = []

    def loadURMdata(self, path):
        parser = ParserURM()
        parser.generateURMfromFile(path)
        self.userList = parser.getUserList()
        print("[DataSplitter] Loaded user list (complete)")
        self.itemList = parser.getItemList()
        print("[DataSplitter] Loaded item list (complete)")
        self.userList_unique = parser.getUserList_unique()
        print("[DataSplitter] Loaded user list (unique)")
        self.itemList_unique = parser.getItemList_unique()
        print("[DataSplitter] Loaded user list (unique)")

    def splitData(self):
        # from the complete lists we have to exclude one random interaction per each user
        # this means that we have to collect the couple (userId, ItemId) and add it to the test URM
        print("[DataSplitter] Splitting data . . .")
        counter = 0
        print("Number of unique users: ", len(self.userList_unique))
        for uniqueUserIndex in range(len(self.userList_unique) - 1):
            startIndex = self.userList.index(self.userList_unique[uniqueUserIndex])
            endIndex = self.userList.index(self.userList_unique[uniqueUserIndex + 1])

            if uniqueUserIndex % 3000 == 0:
                print("     Splitting status: ", counter*1000, "users computed")
                counter = counter + 1

            elementToRemove = random.randint(startIndex, endIndex - 1)

            self.userList_testSet.append(self.userList.pop(elementToRemove))
            self.itemList_testSet.append(self.itemList.pop(elementToRemove))

        if(len(self.userList_testSet) == len(self.itemList_testSet)):
            print("[DataSplitter] Data splitted correctly")

            dim = len(self.userList_testSet)
            ones = []
            for i in range (dim):
                ones.append(1.0)

            print(ones[:10])
            self.URM_test = sps.coo_matrix((ones, (self.userList_testSet, self.itemList_testSet))).tocsr()
            sps.save_npz("data/competition/URM_test.npz", self.URM_test, compressed=True)
            print("[DataSplitter] URM_test created and stored correctly in: data/competition/URM_test.npz")

            del ones[:]
            for i in self.userList:
                ones.append(1.0)
            self.URM_train = sps.coo_matrix((ones, (self.userList, self.itemList))).tocsr()
            sps.save_npz("data/competition/URM_train.npz", self.URM_train, compressed=True)
            print("[DataSplitter] URM_train created and stored correctly in: data/competition/URM_test.npz")

        else:
            print("[DataSplitter : ERROR] The number lenght of the user list for the test set is different from the len of the item list!")

        return


    def userList_toString(self, to):
        print("UserList[0:", to, "]: ", self.userList[:to])

    def itemList_toString(self, to):
        print("ItemList[0:", to, "]: ", self.itemList[:to])

    def userListUnique_toString(self, to):
        print("UserList_unique[0:", to, "]: ", self.userList_unique[:to])

    def itemListUnique_toString(self, to):
        print("ItemList_unique[0:", to, "]: ", self.itemList_unique[:to])

    def userListTest_toString(self, to):
        print("UserListTest[0:", to, "]: ", self.userList_testSet[:to])

    def itemListTest_toString(self, to):
        print("ItemListTest[0:", to, "]: ", self.itemList_testSet[:to])
