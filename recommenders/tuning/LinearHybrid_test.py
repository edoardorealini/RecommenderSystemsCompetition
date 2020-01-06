from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

from recommenders.LinearHybridRecommender import LinearHybridRecommender

from DataUtils.dataLoader import *
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Notebooks_utils.data_splitter import train_test_holdout

from DataUtils.ouputGenerator import create_output_coldUsers_Age

import numpy as np

# Data splitting
resplit_data = False  # DO NOT TOUCH

# Train information
test_model_name = "model_URM_all"  # use different name when training with different parameters
test_model_name_elastic = "new_split_fun"  # Change only if retrain in train test ElasticNet, otherwise don't touch here
retrain = True  # to use in case of change in default parameters of each recommender class

# Evaluation
evaluate_hybrid = False

# Parameter search
search_parameters_random = False
iterations = 10
tuning_log_name = "weights_seek_2randararys"

# Output generation
use_URM_all = True
create_output = True
output_file_name = "06_01_superTuna"

ItemCFKNN_weight = 1.878350
RP3beta_weight = 1.638739
SLIMElasticNet_weight = 1.097934
ItemCBF_weight = 0.271109
UserCFKNN_weight = 0.301294
SLIMCython_weight = 1.500200

URM_train, URM_test = load_data_split(0)

URM_all = load_URM_all()

ICM_all = load_ICM()

if resplit_data:
    '''
    splitter = datasetSplitter()
    splitter.loadURMdata("C:/Users/Utente/Desktop/RecSys-Competition-2019/data/competition/data_train.csv")
    splitter.splitDataBetter()
    '''
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.8)

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
    ItemCFKNN.fit(topK=39, shrink=28)
    ItemCFKNN.save_model(name=test_model_name)

    RP3beta.fit(topK=44)
    RP3beta.save_model(name=test_model_name)

    SLIMElasticNet.load_model(name=test_model_name_elastic)
    # SLIMElasticNet.fit(topK=100)
    # SLIMElasticNet.save_model(name=test_model_name)

    ItemCBF.fit(topK=114, shrink=44)
    ItemCBF.save_model(name=test_model_name)

    UserCFKNN.fit(topK=500, shrink=0.18)
    UserCFKNN.save_model(name=test_model_name)

    # SLIMCython.load_model(name=test_model_name)
    SLIMCython.fit(epochs=500, topK=200)
    SLIMCython.save_model(name=test_model_name)

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
