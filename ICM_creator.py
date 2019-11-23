from sklearn import preprocessing
import numpy as np
import scipy.sparse as sps

# The main idea is to build an ICM using all the data given about the items
# we need to merge the information contained into 3 different files
# data_ICM_asset.csv contains a series of tags for each item (one tag per item)
# data_ICM_price.cvs contains a tag that identifies the price for each item in the item list
# data_ICM_sub_class.csv contains for each item the correspondency with a sub class

#open the 3 files containing the data
ICM_asset_file = open("data/competition/data_ICM_asset.csv", 'r')
ICM_price_file = open("data/competition/data_ICM_price.csv", 'r')
ICM_subclass_file = open("data/competition/data_ICM_sub_class.csv", 'r')

def rowSplit_subclassData(rowString):
    split = rowString.split(",")

    split[0] = int(split[0])    # list of items
    split[1] = int(split[1])    # list of features
    split[2] = int(split[2])    # ones of the matrix

    result = tuple(split)
    return result

def rowSplit_assetData(rowString):
    split = rowString.split(",")

    split[0] = int(split[0])    # item
    split[1] = int(split[1])    # useless
    split[2] = float(split[2])  # asset.tag is a float elem (eazyer)

    result = tuple(split)
    return result

def rowSplit_priceData(rowString):
    split = rowString.split(",")

    split[0] = int(split[0])    # item
    split[1] = int(split[1])    # useless
    split[2] = float(split[2])  # price.tag is a float elem (eazyer)

    result = tuple(split)
    return result

ICM_subclass_tuples = []
for line in ICM_subclass_file:
    ICM_subclass_tuples.append(rowSplit_assetData(line))

itemList_subclass, tagList_subclass, subClassesOnes = zip(*ICM_subclass_tuples)

print(itemList_subclass[:10])
print(tagList_subclass[:10])
print(subClassesOnes[:10])

#now the idea is to extract the remaining tag elements from the other files and merge
# them together into one single matrix that is the  ICM

# reading the data from the files !

ICM_asset_tuples = []
ICM_price_tuples = []

for line in ICM_asset_file:
    ICM_asset_tuples.append(rowSplit_assetData(line))

for line in ICM_price_file:
    ICM_price_tuples.append(rowSplit_priceData(line))

itemList_asset, useless, tagList_asset = zip(*ICM_asset_tuples)
itemList_price, useless, tagList_price = zip(*ICM_price_tuples)

print(tagList_asset[:10])
print(tagList_price[:10])

# now i want to make uniform the list of tags, there cannot be floats and ints, ALL INTS !

allTagsList = []
list_asset = list(tagList_asset)
list_price = list(tagList_price)
list_subclass = list(tagList_subclass)

allTagsList = list_asset + list_subclass  + list_price

le = preprocessing.LabelEncoder()
le.fit(allTagsList)
allTagsList = le.transform(allTagsList)

print(allTagsList[:10])

itemList_complete = itemList_asset + itemList_subclass + itemList_price

n_items = len(itemList_subclass)
n_tags = max(allTagsList) + 1

ICM_shape = (n_items, n_tags)

ones = np.ones(len(allTagsList))
ICM_all = sps.coo_matrix((ones, (itemList_complete, allTagsList)), shape = ICM_shape)
ICM_all = ICM_all.tocsr()

print(ICM_all)

sps.save_npz('data/competition/sparse_ICM.npz', ICM_all, compressed=True)
print("Matrix saved in sparse_ICM.npz")

from ICM_exploration import makeICMStats

makeICMStats(ICM_all)