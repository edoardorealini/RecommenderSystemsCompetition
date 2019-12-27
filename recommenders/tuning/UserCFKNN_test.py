import scipy.sparse as sps
import numpy as np
from Notebooks_utils.evaluation_function import evaluate_algorithm, evaluate_algorithm_coldUsers, evaluate_algorithm_original
from recommenders import UserCFKNNRecommender
from time import time

URM_all = sps.load_npz('data/competition/sparse_URM.npz')
print("URM correctly loaded from file: data/competition/sparse_URM.npz")
URM_all = URM_all.tocsr()

URM_test = sps.load_npz('data/competition/URM_test.npz')
print("URM_test correctly loaded from file: data/competition/URM_test.npz")
URM_test = URM_test.tocsr()

URM_train = sps.load_npz('data/competition/URM_train.npz')
print("URM_train correctly loaded from file: data/competition/URM_train.npz")
URM_train = URM_train.tocsr()

recommender = UserCFKNNRecommender(URM_train)

'''

start_time = time.time()
recommender.fit(shrink=10, topK=750)
end_time = time.time()
print("Fit time: {:.2f} sec".format(end_time-start_time))

create_output("UserCFKNN", recommender)

sh = 20
k = 10

start_time = time.time()
recommender.fit(shrink=sh, topK=k)
end_time = time.time()
print("Fit time: {:.2f} sec".format(end_time-start_time))

start_time = time.time()
print("Evaluating with shrink value = {} and topK = {}".format(sh, k))
result = evaluate_algorithm_original(URM_test, recommender, at=10)
end_time = time.time()
print("Evaluation time: {:.2f} minutes".format((end_time-start_time)/60))
'''

# evaluation done on half the user pool to speed the things up
# Best Shrink value is 2 with one single split evaluation

shrink_values = [2, 3, 5, 7, 8, 9]
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