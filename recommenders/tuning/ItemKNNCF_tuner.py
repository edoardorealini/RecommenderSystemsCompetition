from bayes_opt import BayesianOptimization
import time

from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

from DataUtils.dataLoader import *

from Base.Evaluation.Evaluator import EvaluatorHoldout
from Notebooks_utils.data_splitter import train_test_holdout

from mailer import mail_to


URM_all = load_URM_all()
URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.8)

evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])

def run(topK, shrink):

    ItemCF = ItemKNNCFRecommender(URM_train)
    ItemCF.fit(topK=topK, shrink=shrink, normalize=True, similarity="jaccard")

    result_dict, result_string = evaluator.evaluateRecommender(ItemCF)
    map = result_dict[10]["MAP"]

    return map


if __name__ == '__main__':
    # Bounded region of parameter space
    pbounds = {     
                    'shrink': (0, 200),
                    'topK': (0,1000)
              }

    optimizer = BayesianOptimization(
        f=run,
        pbounds=pbounds,
        verbose=2  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )

    start_time = time.time()

    optimizer.maximize(
        init_points=10,  # random steps
        n_iter=20      # iterations after random steps
    )

    end_time = time.time()

    print(optimizer.max)
    print("Elapsed time = {} minutes".format((end_time-start_time)/60))

    elapsed_time = ((end_time-start_time)/60)

    mail_to(toaddr="realini.edoardo@gmail.com",
            subject="[paramenterTuning] ItemCFKNN parameter search completed", 
            text="\nThe results are:\n" + str(optimizer.max) + "\n Elapsed time: " + str(elapsed_time) + " minutes")
