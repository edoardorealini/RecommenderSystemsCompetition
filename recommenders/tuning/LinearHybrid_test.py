from DataUtils.ouputGenerator import *
from Notebooks_utils.evaluation_function import evaluate_algorithm_original
from recommenders.ItemCFKNNRecommender import ItemCFKNNRecommender
from recommenders.RP3betaGraphBased import RP3betaRecommender
from recommenders.SLIM_ElasticNet import SLIMElasticNetRecommender
from recommenders.LinearHybridRecommender import LinearHybridRecommender
from DataUtils.ouputGenerator import create_output_coldUsers_Age

resplit_data = False

test_model_name = "test3"  # use different name when training with different parameters
test_model_name_elastic = "test2_URM_train"
retrain = True  # to use in case of change in default parameters of each recommender class

evaluate_hybrid = False

use_URM_all = True
create_output = True
output_file_name = "linear_hybrid_test1_28_12"

a = 1.2  # alpha value, wight for ItemCFKNN
b = 0.7  # beta value, weight for RP3beta
g = 0.5  # gamma value, weight for SLIMElasticNet

if resplit_data:
    from DataUtils.datasetSplitter import datasetSplitter
    import scipy.sparse as sps

    splitter = datasetSplitter()
    splitter.loadURMdata("C:/Users/Utente/Desktop/RecSys-Competition-2019/data/competition/data_train.csv")
    splitter.splitDataBetter()

URM_all = sps.load_npz('C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/sparse_URM.npz')
print("URM correctly loaded from file: data/competition/sparse_URM.npz")
URM_all = URM_all.tocsr()

URM_test = sps.load_npz('C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/URM_test.npz')
print("URM_test correctly loaded from file: data/competition/URM_test.npz")
URM_test = URM_test.tocsr()

URM_train = sps.load_npz('C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/URM_train.npz')
print("URM_train correctly loaded from file: data/competition/URM_train.npz")
URM_train = URM_train.tocsr()

if use_URM_all:
    print("[LinearHybrid_test] Creating output, training algoritms on URM_all")
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

    ItemCFKNN_similarity = ItemCFKNN.get_model()
    RP3beta_similarity = RP3beta.get_model()
    SLIMElasticNet_similarity = SLIMElasticNet.get_model()


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