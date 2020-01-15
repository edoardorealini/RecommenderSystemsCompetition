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

from DataUtils.ouputGenerator import *
from recommenders.output.cold_finder import find_cold_in_output

import numpy as np

# Data splitting
resplit_data = False  # DO NOT TOUCH

# Train information
test_model_name = "new_split_fun"  # use different name when training with different parameters
test_model_name_elastic = "new_split_fun"  # Change only if retrain in train test ElasticNet, otherwise don't touch here
retrain = False  # to use in case of change in default parameters of each recommender class
split_n = 0

# Evaluation
evaluate_hybrid = False

# Parameter search
search_parameters_random = False
iterations = 10
tuning_log_name = "weights_seek_2randararys"

# Output generation
create_output = True
normalize_Scores = True
use_URM_all = True

cbf_on_tails = False
cbf_tail_length = 3

ItemCBF_plusWeight = 8.5
n_interactions = 2  # number of interactions to identify users with (for pseudo cold user)

output_file_name = "15_01_hyperTuned2_pseudoCold+8.5"


'''
{'target': 0.04912442739522255,  
'params': {
elastic_weight': 0.46804532624957473, 
'item_cbf_weight': 1.152295146424421, 
'item_cf_weight': 4.718328516704366, 
'rp3_weight': 4.838701700476661, 
'slim_bpr_weight': 0.2696235904011519, 
'user_cf_weight': 4.7815202122250815
}} 


SLIMElasticNet_weight = 0.46804532624957473
ItemCBF_weight = 1.152295146424421
ItemCFKNN_weight = 4.718328516704366
RP3beta_weight = 4.838701700476661
SLIMCython_weight = 0.2696235904011519
UserCFKNN_weight = 4.7815202122250815

elastic_weight=2.096, 
item_cbf_weight=5.851, 
item_cf_weight=4.934,                  
rp3_weight=4.541, 
slim_weight=1.169, 
user_cf_weight=0.06014

mf_weight=0.3342, 

SLIMElasticNet_weight = 2.081129602840116
ItemCBF_weight = 1.9728545676690088
ItemCFKNN_weight = 4.428852626906557
RP3beta_weight = 5.152671308158803
SLIMCython_weight = 0.41541262709817944
UserCFKNN_weight = 4.980983458429952
'''

SLIMElasticNet_weight = 2.081129602840116
ItemCBF_weight = 1.9728545676690088
ItemCFKNN_weight = 4.428852626906557
RP3beta_weight = 5.152671308158803
SLIMCython_weight = 0.41541262709817944
UserCFKNN_weight = 4.980983458429952

URM_train, URM_test = load_data_split(split_number=split_n)
URM_all = load_URM_all()

if resplit_data:
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.8)

ICM_all = load_ICM()
# The error was: loading an older version of the ICM
evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])

if use_URM_all:
    print("[LinearHybrid_test] Creating output, training algorithms on URM_all")
    test_model_name_elastic = "model_URM_all"
    test_model_name = "model_URM_all"
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
    ItemCFKNN.fit(topK=10, shrink=30)
    ItemCFKNN.save_model(name=test_model_name)

    RP3beta.fit(alpha=0.41417, beta=0.04995, min_rating=0, topK=54)
    RP3beta.save_model(name=test_model_name)

    # SLIMElasticNet.load_model(name=test_model_name_elastic)
    SLIMElasticNet.fit(l1_ratio=0.1, alpha=1e-4, topK=100)
    SLIMElasticNet.save_model(name=test_model_name)

    ItemCBF.fit(topK=5, shrink=112, similarity='cosine', normalize=True)
    ItemCBF.save_model(name=test_model_name)

    UserCFKNN.fit(topK=600, shrink=0, similarity='cosine', normalize=True)
    UserCFKNN.save_model(name=test_model_name)

    # SLIMCython.load_model(name=test_model_name)
    SLIMCython.fit(epochs=200, lambda_i=0.0, lambda_j=0.0, learning_rate=0.01, topK=10, sgd_mode='adagrad')
    SLIMCython.save_model(name=test_model_name)

if not retrain:
    print("[LinearHybrid_test] Loading trained models from file, faster approach")
    ItemCFKNN.load_model(name=test_model_name)
    RP3beta.load_model(name=test_model_name)
    SLIMElasticNet.load_model(name=test_model_name)
    ItemCBF.load_model(name=test_model_name)
    UserCFKNN.load_model(name=test_model_name)
    SLIMCython.load_model(name=test_model_name)

if use_URM_all:
    hybrid_CF = LinearHybridRecommender(URM_all, ItemCFKNN, RP3beta, SLIMElasticNet, ItemCBF, UserCFKNN, SLIMCython, normalize=normalize_Scores)
    hybrid_CF.fit(ItemCFKNN_weight, RP3beta_weight, SLIMElasticNet_weight, ItemCBF_weight, UserCFKNN_weight, SLIMCython_weight, retrain_all_algorithms=False)
    hybrid_CBF = LinearHybridRecommender(URM_all, ItemCFKNN, RP3beta, SLIMElasticNet, ItemCBF, UserCFKNN, SLIMCython, normalize=normalize_Scores)
    hybrid_CBF.fit(ItemCFKNN_weight, RP3beta_weight, SLIMElasticNet_weight, ItemCBF_weight + ItemCBF_plusWeight, UserCFKNN_weight, SLIMCython_weight, retrain_all_algorithms=False)

else:
    hybrid_CF = LinearHybridRecommender(URM_train, ItemCFKNN, RP3beta, SLIMElasticNet, ItemCBF, UserCFKNN, SLIMCython, normalize=normalize_Scores)
    hybrid_CF.fit(ItemCFKNN_weight, RP3beta_weight, SLIMElasticNet_weight, ItemCBF_weight, UserCFKNN_weight, SLIMCython_weight, retrain_all_algorithms=False)

if evaluate_hybrid:
    print("[LinearHybrid_test] Evaluating algorithm")
    # evaluate_algorithm_original(URM_test, hybrid, at=10)

    results_dict, results_run_string = evaluator.evaluateRecommender(hybrid_CF)
    print("MAP result = {}".format(results_dict[10]["MAP"]))

if create_output:
    # create_output_coldUsers_Age(output_name=output_file_name, recommender=hybrid_CF, use_cbf_on_tails=cbf_on_tails, cbf_tail_length=cbf_tail_length)
    create_output_superColdMng(output_name=output_file_name, hybrid_CF=hybrid_CF, hybrid_CBF=hybrid_CBF, n_interactions=n_interactions)
    find_cold_in_output(output_file_name)

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
        hybrid_CF.fit(ItemCFKNN_weight, RP3beta_weight, SLIMElasticNet_weight, ItemCBF_weight, UserCFKNN_weight, SLIMCython_weight, retrain_all_algorithms=False)
        dictionary, string = evaluator.evaluateRecommender(hybrid_CF)
        result = "The MAP result for this tuning is:" + str(dictionary[10]["MAP"])
        print(result)
        file_log.write(result + "\n")
        file_log.write("----------------------------------------------------------------------------------------------------")

    file_log.close()
