from DataUtils.ouputGenerator import *
from Notebooks_utils.evaluation_function import evaluate_algorithm_original
from recommenders.ItemCFKNNRecommender import ItemCFKNNRecommender
from recommenders.RP3betaGraphBased import RP3betaRecommender
from recommenders.SLIM_ElasticNet import SLIMElasticNetRecommender
from recommenders.LinearHybridRecommender import LinearHybridRecommender
from DataUtils.ouputGenerator import create_output_coldUsers_Age
from DataUtils.dataLoader import load_all_data
from DataUtils.datasetSplitter import datasetSplitter
import scipy.sparse as sps

import numpy as np

# Data splitting
resplit_data = False

# Train information
test_model_name = "test4"  # use different name when training with different parameters
test_model_name_elastic = "test1_URM_train"
retrain = False  # to use in case of change in default parameters of each recommender class

# Evaluation
evaluate_hybrid = False

# Parameter search
search_parameters_random = True
iterations = 10
tuning_log_name = "random_search_10iter"

# Output generation
use_URM_all = False
create_output = False
output_file_name = "linear_hybrid_test1_28_12"

a = 1.0  # alpha value, wight for ItemCFKNN
b = 0.5  # beta value, weight for RP3beta
g = 0.5  # gamma value, weight for SLIMElasticNet

if resplit_data:
    splitter = datasetSplitter()
    splitter.loadURMdata("C:/Users/Utente/Desktop/RecSys-Competition-2019/data/competition/data_train.csv")
    splitter.splitDataBetter()

URM_all, URM_train, URM_test = load_all_data()

if use_URM_all:
    print("[LinearHybrid_test] Creating output, training algorithms on URM_all")
    test_model_name_elastic = "test2_URM_all"
    ItemCFKNN = ItemCFKNNRecommender(URM_all)
    RP3beta = RP3betaRecommender(URM_all)
    SLIMElasticNet = SLIMElasticNetRecommender(URM_all)

else:
    print("[LinearHybrid_test] Testing Algorithm, training on URM_train")
    ItemCFKNN = ItemCFKNNRecommender(URM_train)
    RP3beta = RP3betaRecommender(URM_train)
    SLIMElasticNet = SLIMElasticNetRecommender(URM_train)

if retrain:
    print("[LinearHybrid_test] Retraining all algorithms, except for SLIM ElasticNet - loading ElasticNet model from file")

    # Note that all the algorithms have decent tuning already as default parameters of fit methods
    ItemCFKNN.fit()
    ItemCFKNN.save_model(name=test_model_name)
    ItemCFKNN_similarity = ItemCFKNN.get_model()

    RP3beta.fit()
    RP3beta.save_model(name=test_model_name)
    RP3beta_similarity = RP3beta.get_model()

    SLIMElasticNet.load_model(name=test_model_name_elastic)
    SLIMElasticNet_similarity = SLIMElasticNet.get_model()

if not retrain:
    print("[LinearHybrid_test] Loading trained models from file, faster approach")
    ItemCFKNN.load_model(name=test_model_name)
    RP3beta.load_model(name=test_model_name)
    SLIMElasticNet.load_model(name=test_model_name_elastic)

if use_URM_all:
    hybrid = LinearHybridRecommender(URM_all, ItemCFKNN, RP3beta, SLIMElasticNet)
    hybrid.fit(alpha=a, beta=b, gamma=g)
else:
    hybrid = LinearHybridRecommender(URM_train, ItemCFKNN, RP3beta, SLIMElasticNet)
    hybrid.fit(alpha=a, beta=b, gamma=g)

if evaluate_hybrid:
    print("[LinearHybrid_test] Evaluating algorithm")
    evaluate_algorithm_original(URM_test, hybrid, at=10)

if create_output:
    create_output_coldUsers_Age(output_name=output_file_name, recommender=hybrid)

if search_parameters_random:
    alpha_values = []
    beta_values = []
    gamma_values = []

    for n in range(5, 21):
        alpha_values.append(n/10)

    for n in range(1, 11):
        beta_values.append(n/10)
        gamma_values.append(n/10)

    file_log = open("C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/output/parameter_tuning/" + tuning_log_name + ".txt", "w")

    for n_iter in range(iterations + 1):
        alpha = np.random.choice(alpha_values, 1)
        beta = np.random.choice(beta_values, 1)
        gamma = np.random.choice(gamma_values, 1)

        file_log.write("\n")
        file_log.write(str(n_iter) + "Fitting hybrid with parameters: alpha={}, beta={}, gamma={}".format(alpha, beta, gamma))
        hybrid.fit(alpha, beta, gamma)
        file_log.write(str(evaluate_algorithm_original(URM_test, hybrid, at=10)))
        file_log.write("----------------------------------------------------------------------------------------------------")

    file_log.close()
