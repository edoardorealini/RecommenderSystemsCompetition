from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np
from DataUtils.ouputGenerator import *
from Notebooks_utils.evaluation_function import evaluate_algorithm_coldUsers, evaluate_algorithm_original
from recommenders import TopPopRecommender as tp
from recommenders.ItemCFKNNRecommender import ItemCFKNNRecommender
from recommenders.RP3betaGraphBased import RP3betaRecommender
from recommenders.SLIM_ElasticNet import SLIMElasticNetRecommender
from recommenders.HybridRecommender import HybridRecommender

URM_all = sps.load_npz('C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/sparse_URM.npz')
print("URM correctly loaded from file: data/competition/sparse_URM.npz")
URM_all = URM_all.tocsr()

URM_test = sps.load_npz('C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/URM_test.npz')
print("URM_test correctly loaded from file: data/competition/URM_test.npz")
URM_test = URM_test.tocsr()

URM_train = sps.load_npz('C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/URM_train.npz')
print("URM_train correctly loaded from file: data/competition/URM_train.npz")
URM_train = URM_train.tocsr()

# URM_train = URM_all
test_model_name = "test2"
retrain = False
use_URM_all = False

a = 0.5  # alpha value, wight for ItemCFKNN
b = 0.3  # beta value, weight for RP3beta
g = 0.2  # gamma value, weight for SLIMElasticNet

if use_URM_all:
    ItemCFKNN = ItemCFKNNRecommender(URM_all)
    RP3beta = RP3betaRecommender(URM_all)
    SLIMElasticNet = SLIMElasticNetRecommender(URM_all)

else:
    ItemCFKNN = ItemCFKNNRecommender(URM_train)
    RP3beta = RP3betaRecommender(URM_train)
    SLIMElasticNet = SLIMElasticNetRecommender(URM_train)

if retrain:
    print("[Hybrid_test] Retraining all algorithms, except for SLIM ElasticNet - loading ElasticNet model from file")

    # Note that all the algorithms have decent tuning already as default parameters of fit methods
    ItemCFKNN.fit()
    ItemCFKNN.save_model(name=test_model_name)
    ItemCFKNN_similarity = ItemCFKNN.get_model()

    RP3beta.fit()
    RP3beta.save_model(name=test_model_name)
    RP3beta_similarity = RP3beta.get_model()

    SLIMElasticNet.load_model(name=test_model_name)
    SLIMElasticNet_similarity = SLIMElasticNet.get_model()

if not retrain:
    ItemCFKNN.load_model(name=test_model_name)
    RP3beta.load_model(name=test_model_name)
    SLIMElasticNet.load_model(name=test_model_name)

    ItemCFKNN_similarity = ItemCFKNN.get_model()
    RP3beta_similarity = RP3beta.get_model()
    SLIMElasticNet_similarity = SLIMElasticNet.get_model()


hybrid = HybridRecommender(URM_train, ItemCFKNN_similarity, RP3beta_similarity, SLIMElasticNet_similarity)
hybrid.fit(alpha=a, beta=b, gamma=g)

hybrid.save_model(name=test_model_name)

evaluate_algorithm_original(URM_test, hybrid, at=10)