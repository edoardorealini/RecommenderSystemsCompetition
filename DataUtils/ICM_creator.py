from sklearn import preprocessing
import numpy as np
import scipy.sparse as sps
from DataUtils.ParserICM import icm_all_builder

# The main idea is to build an ICM using all the data given about the items
# we need to merge the information contained into 3 different files
# data_ICM_asset.csv contains a series of tags for each item (one tag per item)
# data_ICM_price.cvs contains a tag that identifies the price for each item in the item list
# data_ICM_sub_class.csv contains for each item the correspondency with a sub class

#open the 3 files containing the data

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

ICM_asset_file = open("data/competition/data_ICM_asset.csv", 'r')
ICM_price_file = open("data/competition/data_ICM_price.csv", 'r')
ICM_subclass_file = open("data/competition/data_ICM_sub_class.csv", 'r')
urm_all = sps.load_npz("data/competition/sparse_URM.npz")

ICM_subclass_tuples = []
ICM_asset_tuples = []
ICM_price_tuples = []

for line in ICM_subclass_file:
    ICM_subclass_tuples.append(rowSplit_assetData(line))

for line in ICM_asset_file:
    ICM_asset_tuples.append(rowSplit_assetData(line))

for line in ICM_price_file:
    ICM_price_tuples.append(rowSplit_priceData(line))

ICM_all = icm_all_builder(urm_all, ICM_asset_tuples, ICM_price_tuples, ICM_subclass_tuples)

sps.save_npz('data/competition/sparse_ICM.npz', ICM_all, compressed=True)
sps.save_npz('C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/sparse_ICM.npz', ICM_all, compressed=True)
print("Matrix saved in sparse_ICM.npz")

#from DataUtils.ICM_exploration import makeICMStats
#makeICMStats(ICM_all)