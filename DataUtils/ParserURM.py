# class for loading, parsing and splitting the URM given in input by the professor
import scipy.sparse as sps


class ParserURM():
    def __init__(self):
        self.URM_path = ""
        self.userList = []
        self.itemList = []
        self.ratingsList = []


    def rowSplit(self, rowString):
        split = rowString.split(",")
        split[0] = int(split[0])    # User
        split[1] = int(split[1])    # Item
        split[2] = float(split[2])  # Rating, Implicit
        result = tuple(split)
        return result

    def createLists(self, tuples):
        userList, itemList, ratingsList = zip(*tuples)
        self.userList = list(userList)
        self.itemList = list(itemList)
        self.ratingsList = list(ratingsList)
        print("[URM_parser]: lists from file created and exported")
        return self.userList, self.itemList, self.ratingsList

    # Returns the URM in the generic coo format
    def generateURMfromFile(self, path):
        URM_file = open(path, 'r')
        URM_tuples = []
        print("[URM_parser]: splitting rows . . .")
        for row in URM_file:
            URM_tuples.append(self.rowSplit(row))

        self.userList, self.itemList, self.ratingsList = self.createLists(URM_tuples)
        self.URM = sps.coo_matrix((self.ratingsList, (self.userList, self.itemList)))
        print("[URM_parser]: URM correctly loaded, use getters to get the correct format and lists")

    def getUserList(self):
        return self.userList

    def getItemList(self):
        return self.itemList

    def getRatingsList(self):
        return self.ratingsList

    def getUserList_unique(self):
        return list(set(self.userList))

    def getItemList_unique(self):
        return list(set(self.itemList))

    def getRatingsList_unique(self):
        return list(set(self.ratingsList))

    # Returns the URM in native coo formats of scipy
    def getURM(self):
        return self.URM

    # Returns the URM in the CSR format
    def getURM_csr(self):
        self.URM_csr = self.URM.tocsr()
        print("[URM_parser]: URM converted into CSR format")
        return self.URM_csr

    def getURM_csc(self):
        self.URM_csc = self.URM.tocsc()
        print("[URM_parser]: URM converted into CSC format")
        return self.URM_csc

    # Saves the URM in coo format in file
    def saveURMtoFile(self, path, filename):
        sps.save_npz(path + filename, self.URM, compressed=True)
        print("[URM_parser]: File " + filename + " correctly saved in path: " + path)

    def getURMfromFile(self, path):
        self.URM = sps.load_npz(path)
        print("[URM_parser]: loaded URM from path: " + path)