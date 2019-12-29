from DataUtils.dataLoader import *
from DataUtils.datasetSplitter import datasetSplitter

from recommenders.ItemCFKNNRecommender import ItemCFKNNRecommender
from recommenders.RP3betaGraphBased import RP3betaRecommender
from recommenders.SLIM_ElasticNet import SLIMElasticNetRecommender
from recommenders.LinearHybridRecommender import LinearHybridRecommender
from recommenders.CBFRecommender import ItemCBFKNNRecommender
from recommenders.UserCFKNNRecommender import UserCFKNNRecommender

from Notebooks_utils.evaluation_function import evaluate_algorithm_original

from DataUtils.ouputGenerator import *
from DataUtils.ouputGenerator import create_output_coldUsers_Age

import numpy as np

# Data splitting
resplit_data = False  # DO NOT TOUCH

# Train information
test_model_name = "test7.3_URM_train"  # use different name when training with different parameters
test_model_name_elastic = "test3_URM_train"  # Change only if retrain in train test ElasticNet, otherwise don't touch here
retrain = False  # to use in case of change in default parameters of each recommender class

# Evaluation
evaluate_hybrid = True

# Parameter search
search_parameters_random = False
iterations = 10
tuning_log_name = "random_search_10iter"

# Output generation
use_URM_all = False
create_output = False
output_file_name = "lincomb_hyb_5algs_29_12"

ItemCFKNN_weight = 2
RP3beta_weight = 1.0
SLIMElasticNet_weight = 1.0
ItemCBF_weight = 1.0
UserCFKNN_weight = 1.0

if resplit_data:
    splitter = datasetSplitter()
    splitter.loadURMdata("C:/Users/Utente/Desktop/RecSys-Competition-2019/data/competition/data_train.csv")
    splitter.splitDataBetter()

URM_all, URM_train, URM_test = load_all_data()
ICM_all = load_ICM()

if use_URM_all:
    print("[LinearHybrid_test] Creating output, training algorithms on URM_all")
    test_model_name_elastic = "test2_URM_all"
    ItemCFKNN = ItemCFKNNRecommender(URM_all)
    RP3beta = RP3betaRecommender(URM_all)
    SLIMElasticNet = SLIMElasticNetRecommender(URM_all)
    ItemCBF = ItemCBFKNNRecommender(URM_all, ICM=ICM_all)
    UserCFKNN = UserCFKNNRecommender(URM_all)

else:
    print("[LinearHybrid_test] Testing Algorithm, training on URM_train")
    ItemCFKNN = ItemCFKNNRecommender(URM_train)
    RP3beta = RP3betaRecommender(URM_train)
    SLIMElasticNet = SLIMElasticNetRecommender(URM_train)
    ItemCBF = ItemCBFKNNRecommender(URM_train, ICM=ICM_all)
    UserCFKNN = UserCFKNNRecommender(URM_train)

if retrain:
    print("[LinearHybrid_test] Retraining all algorithms, except for SLIM ElasticNet - loading ElasticNet model from file")

    # Note that all the algorithms have decent tuning already as default parameters of fit methods
    ItemCFKNN.fit()
    ItemCFKNN.save_model(name=test_model_name)

    RP3beta.fit()
    RP3beta.save_model(name=test_model_name)

    SLIMElasticNet.load_model(name=test_model_name_elastic)

    ItemCBF.fit()
    ItemCBF.save_model(name=test_model_name)

    UserCFKNN.fit()
    UserCFKNN.save_model(name=test_model_name)

if not retrain:
    print("[LinearHybrid_test] Loading trained models from file, faster approach")
    ItemCFKNN.load_model(name=test_model_name)
    RP3beta.load_model(name=test_model_name)
    SLIMElasticNet.load_model(name=test_model_name_elastic)
    ItemCBF.load_model(name=test_model_name)
    UserCFKNN.load_model(name=test_model_name)

if use_URM_all:
    hybrid = LinearHybridRecommender(URM_all, ItemCFKNN, RP3beta, SLIMElasticNet, ItemCBF, UserCFKNN)
    hybrid.fit(ItemCFKNN_weight, RP3beta_weight, SLIMElasticNet_weight, ItemCBF_weight, UserCFKNN_weight, retrain_all_algorithms=False)
else:
    hybrid = LinearHybridRecommender(URM_train, ItemCFKNN, RP3beta, SLIMElasticNet, ItemCBF, UserCFKNN)
    hybrid.fit(ItemCFKNN_weight, RP3beta_weight, SLIMElasticNet_weight, ItemCBF_weight, UserCFKNN_weight, retrain_all_algorithms=False)

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
