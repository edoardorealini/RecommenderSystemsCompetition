#load information
import scipy.sparse as sps
from recommenders.old_Algorithms.ItemCFKNNRecommender import ItemCFKNNRecommender
from recommenders.old_Algorithms.UserBasedRecommender import *
from DataUtils.datasetSplitter import getColdUsers

URM_all = sps.load_npz('data/competition/sparse_URM.npz')
print("URM correctly loaded from file: data/competition/sparse_URM.npz")
URM_all = URM_all.tocsr()

URM_test = sps.load_npz('data/competition/URM_test.npz')
print("URM_test correctly loaded from file: data/competition/URM_test.npz")
URM_test = URM_test.tocsr()

URM_train = sps.load_npz('data/competition/URM_train.npz')
print("URM_train correctly loaded from file: data/competition/URM_train.npz")
URM_train = URM_train.tocsr()

UCM_age = sps.load_npz("data/competition/sparse_UCM_age.npz")
print("UCM_age correctly loaded from file: data/competition/sparse_UCM_age.npz")
UCM_age = UCM_age.tocsr()

UCM_region = sps.load_npz("data/competition/sparse_UCM_region.npz")
print("UCM_region correctly loaded from file: data/competition/sparse_UCM_region.npz")
UCM_region = UCM_region.tocsr()

CF_recommender = ItemCFKNNRecommender(URM_all)
print("[CFRecommender] Fitting . . .")
CF_recommender.fit(shrink=25, topK=50, similarity="jaccard")

userBased_recommender = UserBasedRecommender(UCM_age, CF_recommender)
userBased_recommender.fit()

coldUserList = getColdUsers()
recommendations = []

for user in coldUserList:
    recommendations.append((user, userBased_recommender.recommend(user_id=user, at=10)))

## ERRORE: ogni utente ha una sola età: è impossibile trovare un utente simile basandosi solo su questo dato!
print(recommendations[:10])