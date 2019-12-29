import scipy.sparse as sps
import numpy as np
from Notebooks_utils.evaluation_function import evaluate_algorithm_original
from DataUtils.dataLoader import load_all_data
from recommenders.UserCFKNNRecommender import UserCFKNNRecommender
import time

URM_all, URM_train, URM_test = load_all_data()

recommender = UserCFKNNRecommender(URM_train)
recommender.fit()
recommender.save_model(name="test7.3_URM_train")


# Best Shrink value is 0.5 with one single split evaluation
'''
shrink_values = [0.5, 0.7]
shrink_results = []

START = time.time()

for sh in shrink_values:
    print("####################################")
    print("Fitting . . .")
    recommender.fit(shrink=sh, topK=10)
    start_time = time.time()
    print("Evaluating with shrink value = ", sh)
    result = evaluate_algorithm_original(URM_test, recommender, at=10)
    end_time = time.time()
    print("Evaluation time: {:.2f} minutes".format((end_time-start_time)/60))
    shrink_results.append(result["MAP"])

END = time.time()

total_time = (END - START)/60

print("Total time for parameter tuning is {:.2f} minutes".format(total_time))
'''