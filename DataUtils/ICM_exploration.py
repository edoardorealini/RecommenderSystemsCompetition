import numpy as np
import scipy.sparse as sps

URM_all = sps.load_npz('data/competition/sparse_URM.npz')
print("URM correctly loaded from file: data/competition/sparse_URM.npz")

ICM_file = open("data/competition/data_ICM_asset.csv", 'r')
print("ICM_Asset file opened correctly")

def list_ID_stats(ID_list, label):
    min_val = min(ID_list)
    max_val = max(ID_list)
    unique_val = len(set(ID_list))
    missing_val = 1 - unique_val / (max_val - min_val)

    print("{} data, ID: min {}, max {}, unique {}, missig {:.2f} %".format(label, min_val, max_val, unique_val,
                                                                           missing_val * 100))

def rowSplit(rowString):
    split = rowString.split(",")

    split[0] = int(split[0])    # item
    split[1] = int(split[1])    # useless
    split[2] = float(split[2])  # tag is a float (eazyer)

    result = tuple(split)
    return result

ICM_file.seek(0)
ICM_tuples = [] # a ICM tuple is made of (item, feature)

for line in ICM_file:
    ICM_tuples.append(rowSplit(line))

itemList_icm, useless_list, tagList_icm= zip(*ICM_tuples)

itemList_icm = list(itemList_icm)
tagList_icm = list(tagList_icm)
unique_tagList_icm = list(set(tagList_icm))

for _ in range(10):
    print(itemList_icm[_])

for _ in range(10):
    print(tagList_icm[_])

list_ID_stats(itemList_icm, "Items ICM")

n_items = len(itemList_icm)
n_tags = len(tagList_icm) + 1
n_uniquetags = len(set(tagList_icm))

print("Number of tags:" , n_tags)
print("Number of unique tags:" , n_uniquetags)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(tagList_icm)
tagList_icm = le.transform(tagList_icm)
print(tagList_icm[0:10])

n_items = URM_all.shape[1]
n_tags = max(tagList_icm) + 1

ICM_shape = (n_items, n_tags)

ones = np.ones(len(tagList_icm))
ICM_all = sps.coo_matrix((ones, (itemList_icm, tagList_icm)), shape = ICM_shape)
ICM_all = ICM_all.tocsr()

print(ICM_all)

sps.save_npz('data/competition/sparse_ICM.npz', ICM_all, compressed=True)
print("Matrix saved in sparse_ICM.npz")


# STATISTICS

def makeICMStats(ICM_all):
    ICM_all = sps.csr_matrix(ICM_all)
    features_per_item = np.ediff1d(ICM_all.indptr)

    ICM_all = sps.csc_matrix(ICM_all)
    items_per_feature = np.ediff1d(ICM_all.indptr)

    ICM_all = sps.csr_matrix(ICM_all)

    print(features_per_item.shape)
    print(items_per_feature.shape)

    features_per_item = np.sort(features_per_item)
    items_per_feature = np.sort(items_per_feature)

    import matplotlib.pyplot as pyplot

    pyplot.plot(features_per_item, 'ro')
    pyplot.ylabel('Num features ')
    pyplot.xlabel('Item Index')
    pyplot.show()

    pyplot.plot(items_per_feature, 'ro')
    pyplot.ylabel('Num items ')
    pyplot.xlabel('Feature Index')
    pyplot.show()