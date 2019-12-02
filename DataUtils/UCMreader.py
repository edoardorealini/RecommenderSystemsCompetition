import scipy.sparse as sps
from DataUtils.datasetSplitter import rowSplit, createLists


path_age = "./data/competition/data_UCM_age.csv"
path_region = "./data/competition/data_UCM_region.csv"

file_age = open(path_age, "r")
file_region = open(path_region, "r")

age_tuples = []
region_tuples = []

for row in file_age:
    age_tuples.append(rowSplit(row))

for row in file_region:
    region_tuples.append(rowSplit(row))

print(age_tuples[:10])
print(region_tuples[:10])

age_users, age_values, age_ones = createLists(age_tuples)
region_users, region_values, region_ones = createLists(region_tuples)

UCM_age = sps.coo_matrix((age_ones, (age_users, age_values)))
UCM_region = sps.coo_matrix((region_ones, (region_users, region_values)))

UCM_age = UCM_age.tocsr()
UCM_region = UCM_region.tocsr()

sps.save_npz("./data/competition/sparse_UCM_age.npz", UCM_age)
sps.save_npz("./data/competition/sparse_UCM_region.npz", UCM_region)
print("[UCMLoader] Loaded both Age and Region UCM matrix and stored in npz files")

