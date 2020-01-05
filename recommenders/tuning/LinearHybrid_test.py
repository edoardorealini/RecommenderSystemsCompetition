from DataUtils.dataLoader import *
from DataUtils.datasetSplitter import datasetSplitter

'''
from recommenders.ItemCFKNNRecommender import ItemCFKNNRecommender
from recommenders.RP3betaGraphBased import RP3betaRecommender
from recommenders.SLIM_ElasticNet import SLIMElasticNetRecommender
from recommenders.CBFRecommender import ItemCBFKNNRecommender
from recommenders.UserCFKNNRecommender import UserCFKNNRecommender
'''

from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

from recommenders.LinearHybridRecommender import LinearHybridRecommender

from Notebooks_utils.evaluation_function import evaluate_algorithm_original
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Notebooks_utils.data_splitter import train_test_holdout
from DataUtils.dataLoader import *

from DataUtils.ouputGenerator import create_output_coldUsers_Age

import numpy as np

# Data splitting
resplit_data = False  # DO NOT TOUCH

# Train information
test_model_name = "new_split_fun"  # use different name when training with different parameters
test_model_name_elastic = "new_split_fun"  # Change only if retrain in train test ElasticNet, otherwise don't touch here
retrain = False  # to use in case of change in default parameters of each recommender class

# Evaluation
evaluate_hybrid = False

# Parameter search
search_parameters_random = True
iterations = 10
tuning_log_name = "weights_seek_2randararys"

# Output generation
use_URM_all = False
create_output = False
output_file_name = "4rick"

ItemCFKNN_weight = 1.5
RP3beta_weight = 1.2
SLIMElasticNet_weight = 1.2
ItemCBF_weight = 0.3
UserCFKNN_weight = 0.2
SLIMCython_weight = 0.7

URM_all, URM_train, URM_test = load_data_split(0)
ICM_all = load_ICM()

if resplit_data:
    '''
    splitter = datasetSplitter()
    splitter.loadURMdata("C:/Users/Utente/Desktop/RecSys-Competition-2019/data/competition/data_train.csv")
    splitter.splitDataBetter()
    '''
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.8)
    URM_train, URM_validation = train_test_holdout(URM_train, train_perc=0.9)

evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])

if use_URM_all:
    print("[LinearHybrid_test] Creating output, training algorithms on URM_all")
    test_model_name_elastic = "model_URM_all"
    ItemCFKNN = ItemKNNCFRecommender(URM_all)
    RP3beta = RP3betaRecommender(URM_all)
    SLIMElasticNet = SLIMElasticNetRecommender(URM_all)
    ItemCBF = ItemKNNCBFRecommender(URM_all, ICM_all)
    UserCFKNN = UserKNNCFRecommender(URM_all)
    SLIMCython = SLIM_BPR_Cython(URM_all, verbose=False, recompile_cython=False)

else:
    print("[LinearHybrid_test] Testing Algorithm, training on URM_train")
    ItemCFKNN = ItemKNNCFRecommender(URM_train)
    RP3beta = RP3betaRecommender(URM_train)
    SLIMElasticNet = SLIMElasticNetRecommender(URM_train)
    ItemCBF = ItemKNNCBFRecommender(URM_train, ICM_all)
    UserCFKNN = UserKNNCFRecommender(URM_train)
    SLIMCython = SLIM_BPR_Cython(URM_train, verbose=False, recompile_cython=False)


if retrain:
    print("[LinearHybrid_test] Retraining all algorithms, except for SLIM ElasticNet - loading ElasticNet model from file")

    # Note that all the algorithms have decent tuning already as default parameters of fit methods
    ItemCFKNN.fit(topK=10, shrink=30, normalize=True, similarity="jaccard")
    ItemCFKNN.save_model(name=test_model_name)

    RP3beta.fit(alpha=0.41, beta=0.049, min_rating=0, topK=10, implicit=True, normalize_similarity=True)
    RP3beta.save_model(name=test_model_name)

    SLIMElasticNet.load_model(name=test_model_name_elastic)
    # SLIMElasticNet.fit(l1_ratio=0.1, alpha=1e-4)
    # SLIMElasticNet.save_model(name=test_model_name)

    ItemCBF.fit(topK=10, shrink=1, normalize=False, similarity="jaccard")
    ItemCBF.save_model(name=test_model_name)

    UserCFKNN.fit(topK=10, shrink=0.05, normalize=False, similarity="jaccard")
    UserCFKNN.save_model(name=test_model_name)

    SLIMCython.load_model(name=test_model_name)

if not retrain:
    print("[LinearHybrid_test] Loading trained models from file, faster approach")
    ItemCFKNN.load_model(name=test_model_name)
    RP3beta.load_model(name=test_model_name)
    SLIMElasticNet.load_model(name=test_model_name_elastic)
    ItemCBF.load_model(name=test_model_name)
    UserCFKNN.load_model(name=test_model_name)
    SLIMCython.load_model(name=test_model_name)

if use_URM_all:
    hybrid = LinearHybridRecommender(URM_all, ItemCFKNN, RP3beta, SLIMElasticNet, ItemCBF, UserCFKNN, SLIMCython)
    hybrid.fit(ItemCFKNN_weight, RP3beta_weight, SLIMElasticNet_weight, ItemCBF_weight, UserCFKNN_weight, SLIMCython_weight, retrain_all_algorithms=False)
else:
    hybrid = LinearHybridRecommender(URM_train, ItemCFKNN, RP3beta, SLIMElasticNet, ItemCBF, UserCFKNN, SLIMCython)
    hybrid.fit(ItemCFKNN_weight, RP3beta_weight, SLIMElasticNet_weight, ItemCBF_weight, UserCFKNN_weight, SLIMCython_weight, retrain_all_algorithms=False)

if evaluate_hybrid:
    print("[LinearHybrid_test] Evaluating algorithm")
    # evaluate_algorithm_original(URM_test, hybrid, at=10)

    results_dict, results_run_string = evaluator.evaluateRecommender(hybrid)
    print("MAP result = {}".format(results_dict[10]["MAP"]))

if create_output:
    create_output_coldUsers_Age(output_name=output_file_name, recommender=hybrid)

if search_parameters_random:
    high_weights = []
    low_weights = []

    for n in range(10, 21):
        high_weights.append(n/10)

    for n in range(0, 11):
        low_weights.append(n/10)

    file_log = open("C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/output/parameter_tuning/" + tuning_log_name + ".txt", "w")

    for n_iter in range(iterations + 1):
        print("\n[randomWeights_search] Iteration number {}\n".format(n_iter))

        ItemCFKNN_weight = np.random.choice(high_weights, 1)
        RP3beta_weight = np.random.choice(high_weights, 1)
        SLIMElasticNet_weight = np.random.choice(high_weights, 1)
        ItemCBF_weight = np.random.choice(low_weights, 1)
        UserCFKNN_weight = np.random.choice(low_weights, 1)
        SLIMCython_weight = np.random.choice(high_weights, 1)

        file_log.write("\n")
        file_log.write(str(n_iter) + "Fitting with parameters: ItemCFKNN_weight={}, RP3beta_weight={}, SLIMElasticNet_weight={}, ItemCBF_weight={}, UserCFKNN_weight={}, SLIMCython_weight={}"
              .format(ItemCFKNN_weight, RP3beta_weight, SLIMElasticNet_weight, ItemCBF_weight, UserCFKNN_weight, SLIMCython_weight))
        hybrid.fit(ItemCFKNN_weight, RP3beta_weight, SLIMElasticNet_weight, ItemCBF_weight, UserCFKNN_weight, SLIMCython_weight, retrain_all_algorithms=False)
        dictionary, string = evaluator.evaluateRecommender(hybrid)
        result = "The MAP result for this tuning is:" + str(dictionary[10]["MAP"])
        print(result)
        file_log.write(result + "\n")
        file_log.write("----------------------------------------------------------------------------------------------------")

    file_log.close()
