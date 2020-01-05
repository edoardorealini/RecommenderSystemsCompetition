from bayes_opt import BayesianOptimization
import time

from Notebooks_utils.evaluation_function import evaluate_algorithm_original
from recommenders.old_Algorithms.CBFRecommender import ItemCBFKNNRecommender

from DataUtils.dataLoader import *

from mailer import mail_to

split_number = 0


test_model_name = "model_" + str(split_number)  # use different name when training with different parameters


def run(shrink):

    URM_all, URM_train, URM_test = load_all_data()
    ICM_all = load_ICM()

    ItemCBF = ItemCBFKNNRecommender(URM_train, ICM=ICM_all)
    ItemCBF.fit(shrink=shrink)

    return evaluate_algorithm_original(URM_test, ItemCBF)["MAP"]


if __name__ == '__main__':
    # Bounded region of parameter space
    pbounds = { 'shrink' : (0, 100)}

    optimizer = BayesianOptimization(
        f=run,
        pbounds=pbounds,
        verbose=2  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )

    start_time = time.time()

    optimizer.maximize(
        init_points=25,  # random steps
        n_iter=50,      # iterations after random steps
    )

    end_time = time.time()

    print(optimizer.max)
    elapsed_time = ((end_time-start_time)/60)/60 #  elapsed time in hours.
    
    mail_to(toaddr="realini.edoardo@gmail.com",
            subject="[SERVER] Shrink search for ItemCBF completed", 
            text="\nThe results are:\n" + str(optimizer.max) + "\n Elapsed time: " + str(elapsed_time) + " hours")
