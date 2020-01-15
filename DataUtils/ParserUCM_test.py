from DataUtils.ParserUCM import *
from DataUtils.datasetSplitter import rowSplit
from DataUtils.dataLoader import *

# This function takes a value on number of interactions and gives back the corresponding clusterized value
# If interval is given as 10, the totaln number of interactions (max) is divided by ten and all users are clusterized as follows
# inf the n inter is between 2 cluster values, a single number is returned
# This
def interaction_clustering(n_interactions, interval):


    clusterized = 0

    return clusterized


def generate_interaction_tuples(file):
    tuples = []
    one = 1
    user_id = 0

    for line in file:
        line.replace("\n", "")
        itr = int(line)
        tuples.append((user_id, itr, one))
        user_id += 1

    return tuples

URM_all = load_URM_all()

path_age = "./data/competition/data_UCM_age.csv"
path_region = "./data/competition/data_UCM_region.csv"
path_interactions = "./data/competition/users/interactions_per_user.txt"

file_age = open(path_age, "r")
file_region = open(path_region, "r")
file_interactions = open(path_interactions, "r")

age_tuples = []
region_tuples = []
interaction_tuples = []

for row in file_age:
    age_tuples.append(rowSplit(row))

for row in file_region:
    region_tuples.append(rowSplit(row))

interaction_tuples = generate_interaction_tuples(file_interactions)

UCM_all = ucm_all_builder(urm_all=URM_all, ucm_age_tuples=age_tuples, ucm_region_tuples=region_tuples, ucm_interactions_tuples=interaction_tuples)
UCM_all = UCM_all.tocsr()

print("Saving UCM_all on RecSys-Competition-2019/DataUtils/data/competition/UCM_all.npz")
sps.save_npz("C:/Users/Utente/Desktop/RecSys-Competition-2019/DataUtils/data/competition/UCM_all.npz", UCM_all, compressed=True)
print("Saving UCM_all on RecSys-Competition-2019/recommenders/data/competition/UCM_all.npz")
sps.save_npz("C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/UCM_all.npz", UCM_all, compressed=True)