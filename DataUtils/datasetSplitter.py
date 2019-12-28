# With this class we want to split the URM into train, validation and test sets
# The main idea is to split the dataset as follows:
#   - The test set is created starting from the full URM. For each user is collected one interaction and stored in the
#       test set.
import scipy.sparse as sps
import random

from tqdm import tqdm

from DataUtils.ParserURM import ParserURM


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
        self.URM_tuples = []

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
        self.URM_tuples = parser.getTuples()

    def splitData(self):
        # from the complete lists we have to exclude one random interaction per each user
        # this means that we have to collect the couple (userId, ItemId) and add it to the test URM
        print("[DataSplitter] Splitting data . . .")
        counter = 0
        print("Number of unique users: ", len(self.userList_unique))
        for uniqueUserIndex in tqdm(range(len(self.userList_unique) - 1)):
            startIndex = self.userList.index(self.userList_unique[uniqueUserIndex])
            endIndex = self.userList.index(self.userList_unique[uniqueUserIndex + 1])

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
            sps.save_npz("C:/Users/Utente/Desktop/RecSys-Competition-2019/DataUtils/data/competition/URM_test.npz", self.URM_test, compressed=True)
            sps.save_npz("C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/URM_test.npz",
                         self.URM_test, compressed=True)
            print("[DataSplitter] URM_test created and stored correctly in: data/competition/URM_test.npz")

            del ones[:]
            for i in self.userList:
                ones.append(1.0)
            self.URM_train = sps.coo_matrix((ones, (self.userList, self.itemList))).tocsr()
            sps.save_npz("C:/Users/Utente/Desktop/RecSys-Competition-2019/DataUtils/data/competition/URM_train.npz", self.URM_train, compressed=True)
            sps.save_npz("C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/URM_train.npz",
                         self.URM_train, compressed=True)
            print("[DataSplitter] URM_train created and stored correctly in: data/competition/URM_train.npz")

        else:
            print("[DataSplitter : ERROR] The number lenght of the user list for the test set is different from the len of the item list!")

        return

    def splitDataBetter(self):
        # Use tuples directly !
        # Iterate on tuples, same algorithm as before, just avoid using pop on the list, too slow implementation
        # IDEA: for a set of users get one randomly and copy his tuple into a list this generates the test set
        # for the train set, the tuples not copied for the test set should be copied into another list to then form the train set
        tuples = self.URM_tuples

        test_tuples = []
        train_tuples = []

        # save the indexes of the popped items (in order to ignore them when creating the train set!)
        popped_users = []
        # initialization to the first user id in the list
        last_userID = tuples[0][0]
        last_user_index = 0

        # the goal is to overwrite the files URM_test.npz and URM_train.npz
        # this for creates the train split
        print("[Splitter] Splitting . . .")

        for index in range(len(tuples)):
            # t is the tuple !
            # in t[0] is stored the user id, in t[1] is stored the item id
            # each tuple represents an interaction of the user t[0] with the item t[1]
            # in one linear scan,
            user = tuples[index][0]

            if user != last_userID:
                # this case means that we found a user that is not been encountered yet !
                # in index we have the index of the current tuple (the one with the new user)
                # in last_user_index we have the index of the last user
                rand_index = random.randint(last_user_index, index - 1)

                test_tuples.append(tuples[rand_index])
                popped_users.append(rand_index)

                for i in range(last_user_index, index):
                    if i != rand_index:
                        train_tuples.append(tuples[i])

                last_userID = user
                last_user_index = index

        test_users, test_items, test_ratings = zip(*test_tuples)
        train_users, train_items, train_ratings = zip(*train_tuples)

        print("[Splitter] Creating matrices")
        URM_train = sps.coo_matrix((train_ratings, (train_users, train_items))).tocsr()
        URM_test = sps.coo_matrix((test_ratings, (test_users, test_items))).tocsr()

        print("[Splitter] Saving matrices on files")

        sps.save_npz("C:/Users/Utente/Desktop/RecSys-Competition-2019/DataUtils/data/competition/URM_test.npz", URM_test, compressed=True)
        sps.save_npz("C:/Users/Utente/Desktop/RecSys-Competition-2019/DataUtils/data/competition/URM_train.npz", URM_train, compressed=True)

        sps.save_npz("C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/URM_test.npz", URM_test, compressed=True)
        sps.save_npz("C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/URM_train.npz", URM_train, compressed=True)


        return URM_train, URM_test


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



def getColdUsers():
    coldUsers = []
    path = "C:/Users/Utente/Desktop/RecSys-Competition-2019/data/competition/users/userlist_cold.txt"
    file = open(path, 'r')

    for line in file:
        split = line.split()
        coldUsers.append(int(split[0]))

    return coldUsers

def getUserList():
    userlist = []

    path = "C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/users/userlist_output.txt"
    file = open(path, 'r')

    for item in file:
        userlist.append(int(item))

    return userlist

def rowSplit(rowString):
    split = rowString.split(",")
    split[0] = int(split[0])    # User
    split[1] = int(split[1])    # item
    split[2] = float(split[2])  # Rating, Implicit
    result = tuple(split)
    return result

def createLists(tuples):
    userList, itemList, ratingsList = zip(*tuples)
    userList = list(userList)
    itemList = list(itemList)
    ratingsList = list(ratingsList)
    print("[Parser]: lists from file created and exported")
    return userList, itemList, ratingsList