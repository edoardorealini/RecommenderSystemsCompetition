import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as pyplot

def printFirstTenItems(arrayOfItems):
    for i in range(10):
        print(arrayOfItems[i])

URM_all = sps.load_npz('data/competition/sparse_URM.npz')
print("URM correctly loaded from file: data/competition/sparse_URM.npz")

itemPopularity = (URM_all>0).sum(axis=0)
print(type(itemPopularity))
print(itemPopularity)
itemPopularity = np.array(itemPopularity).squeeze()
itemPopularity = np.sort(itemPopularity)

pyplot.plot(itemPopularity, 'ro')
pyplot.ylabel('Num Interactions ')
pyplot.xlabel('Item Index')
pyplot.show()

print("Number of items with zero interactions {}".
      format(np.sum(itemPopularity==0)))

itemPopularityNonzero = itemPopularity[itemPopularity>0]

tenPercent = int(len(itemPopularityNonzero)/10)

print("Average per-item interactions over the whole dataset {:.2f}".
      format(itemPopularityNonzero.mean()))

print("Average per-item interactions for the top 10% popular items {:.2f}".
      format(itemPopularityNonzero[-tenPercent].mean()))

print("Average per-item interactions for the least 10% popular items {:.2f}".
      format(itemPopularityNonzero[:tenPercent].mean()))

userActivity = (URM_all>0).sum(axis=1)
userActivity = np.array(userActivity).squeeze()
userActivity = np.sort(userActivity)

pyplot.plot(userActivity, 'ro')
pyplot.ylabel('Num Interactions ')
pyplot.xlabel('User Index')
pyplot.show()