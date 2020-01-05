from bayes_opt import BayesianOptimization
import time

from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from recommenders.LinearHybridRecommender import LinearHybridRecommender

from DataUtils.dataLoader import *

from DataUtils.ouputGenerator import create_output_coldUsers_Age
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Notebooks_utils.data_splitter import train_test_holdout

test_model_name = "new_split_fun"  # use different name when training with different parameters
test_model_name_elastic = "new_split_fun"
retrain = False

URM_all, URM_train, URM_test = load_data_split(split_number=0)
ICM_all = load_ICM()

evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])

def run(# als_weight,
        item_cbf_weight,
        item_cf_weight,
        elastic_weight,
        rp3_weight,
        # slim_bpr_weight,
        user_cf_weight
        ):

    ItemCFKNN = ItemKNNCFRecommender(URM_train)
    RP3beta = RP3betaRecommender(URM_train)
    SLIMElasticNet = SLIMElasticNetRecommender(URM_train)
    ItemCBF = ItemKNNCBFRecommender(URM_train, ICM_all)
    UserCFKNN = UserKNNCFRecommender(URM_train)

    if retrain:
        ItemCFKNN.fit()
        RP3beta.fit()
        SLIMElasticNet.fit()
        ItemCBF.fit()
        UserCFKNN.fit()

        ItemCFKNN.save_model(name=test_model_name)
        RP3beta.save_model(name=test_model_name)
        SLIMElasticNet.save_model(name=test_model_name_elastic)
        ItemCBF.save_model(name=test_model_name)
        UserCFKNN.save_model(name=test_model_name)

    ItemCFKNN.load_model(name=test_model_name)
    RP3beta.load_model(name=test_model_name)
    SLIMElasticNet.load_model(name=test_model_name_elastic)
    ItemCBF.load_model(name=test_model_name)
    UserCFKNN.load_model(name=test_model_name)

    recommender = LinearHybridRecommender(URM_train, ItemCFKNN, RP3beta, SLIMElasticNet, ItemCBF, UserCFKNN)
    recommender.fit(item_cf_weight, rp3_weight, elastic_weight, item_cbf_weight, user_cf_weight)

    result_dict, result_string = evaluator.evaluateRecommender(recommender)
    map = result_dict[10]["MAP"]

    return map


if __name__ == '__main__':
    # Bounded region of parameter space
    pbounds = {     # 'als_weight': (0, 5),
                    'item_cf_weight': (5, 12),
                    'rp3_weight': (4, 10),
                    'elastic_weight': (3, 7),
                    'item_cbf_weight': (0, 6),
                    # 'slim_bpr_weight': (0, 5),
                    'user_cf_weight': (3, 7)
              }

    optimizer = BayesianOptimization(
        f=run,
        pbounds=pbounds,
        verbose=2  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )

    start_time = time.time()

    optimizer.maximize(
        init_points=5,  # random steps
        n_iter=10,      # iterations after random steps
    )

    end_time = time.time()

    print(optimizer.max)
    print("Elapsed time = {} minutes".format((end_time-start_time)/60))
