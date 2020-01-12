from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

from sklearn import preprocessing

from DataUtils.dataLoader import *

import scipy.sparse as sps

URM_train, URM_test = load_data_split(0)
ICM_all = load_ICM()

recommenders = []

ItemCFKNN = ItemKNNCFRecommender(URM_train)
RP3beta = RP3betaRecommender(URM_train)
SLIMElasticNet = SLIMElasticNetRecommender(URM_train)
ItemCBF = ItemKNNCBFRecommender(URM_train, ICM_all)
UserCFKNN = UserKNNCFRecommender(URM_train)
SLIMCython = SLIM_BPR_Cython(URM_train, verbose=False, recompile_cython=False)

recommenders.append(ItemCFKNN)
recommenders.append(RP3beta)
recommenders.append(SLIMElasticNet)
recommenders.append(ItemCBF)
recommenders.append(UserCFKNN)
recommenders.append(SLIMCython)

max_similarity_values = []
models = []

for recommender in recommenders:
    recommender.load_model(name="model_URM_all")
    models.append(recommender.get_model())
    max_similarity_values.append((recommender.RECOMMENDER_NAME, recommender.get_model().max()))

print("Getting max similarity values in models")
for _ in max_similarity_values:
    print(_)

print("Standardizing models")
norm_models = []
for model in models:
    norm_models.append(preprocessing.normalize(model))

max_similarity_values_norm = []

for i in range(len(norm_models)):
    max_similarity_values_norm.append((recommenders[i].RECOMMENDER_NAME, norm_models[i].max()))

for _ in max_similarity_values_norm:
    print(_)