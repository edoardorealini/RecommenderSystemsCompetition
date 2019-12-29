from bayes_opt import BayesianOptimization
import time

from Notebooks_utils.evaluation_function import evaluate_algorithm_original
from recommenders.CBFRecommender import ItemCBFKNNRecommender
from recommenders.ItemCFKNNRecommender import ItemCFKNNRecommender
from recommenders.RP3betaGraphBased import RP3betaRecommender
from recommenders.SLIM_ElasticNet import SLIMElasticNetRecommender
from recommenders.LinearHybridRecommender import LinearHybridRecommender
from DataUtils.dataLoader import *
from recommenders.UserCFKNNRecommender import UserCFKNNRecommender

test_model_name = "test7.3_URM_train"  # use different name when training with different parameters
test_model_name_elastic = "test3_URM_train"
retrain = False


def run(# als_weight,
        item_cbf_weight,
        item_cf_weight,
        elastic_weight,
        rp3_weight,
        # slim_bpr_weight,
        user_cf_weight
        ):

    URM_all, URM_train, URM_test = load_all_data()
    ICM_all = load_ICM()

    ItemCFKNN = ItemCFKNNRecommender(URM_train)
    RP3beta = RP3betaRecommender(URM_train)
    SLIMElasticNet = SLIMElasticNetRecommender(URM_train)
    ItemCBF = ItemCBFKNNRecommender(URM_train, ICM=ICM_all)
    UserCFKNN = UserCFKNNRecommender(URM_train)

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

    return evaluate_algorithm_original(URM_test, recommender)["MAP"]


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
