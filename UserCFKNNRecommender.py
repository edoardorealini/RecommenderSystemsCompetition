from Notebooks_utils.evaluation_function import evaluate_algorithm
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import scipy.sparse as sps
from urllib.request import urlretrieve
import zipfile, os
import numpy as np
import matplotlib.pyplot as pyplot
import time
from ParserURM import ParserURM
from ouputGenerator import *


class UserCFKNNRecommender(object):

    def __init__(self, URM):
        self.URM = URM

    def fit(self, topK=50, shrink=100, normalize=True, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(self.URM.T, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product

        scores = self.W_sparse[user_id, :].dot(self.URM).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


URM_all = sps.load_npz('data/competition/sparse_URM.npz')
print("URM correctly loaded from file: data/competition/sparse_URM.npz")
URM_all = URM_all.tocsr()

URM_test = sps.load_npz('data/competition/URM_test.npz')
print("URM_test correctly loaded from file: data/competition/URM_test.npz")
URM_test = URM_test.tocsr()

URM_train = sps.load_npz('data/competition/URM_train.npz')
print("URM_train correctly loaded from file: data/competition/URM_train.npz")
URM_train = URM_train.tocsr()

ICM_all = sps.load_npz('data/competition/sparse_ICM.npz')
print("ICM_all correctly loaded from path: data/competition/sparse_ICM.npz")
ICM_all = ICM_all.tocsr()

parser = ParserURM()
URM_path = "data/competition/data_train.csv"
parser.generateURMfromFile(URM_path)

userList = parser.getUserList_unique()

recommender = UserCFKNNRecommender(URM_train)

start_time = time.time()
recommender.fit(shrink=10, topK=750)
end_time = time.time()
print("Fit time: {:.2f} sec".format(end_time-start_time))

create_output("UserCFKNN", recommender)

'''
sh = 50
k = 100

start_time = time.time()
recommender.fit(shrink=sh, topK=k)
end_time = time.time()
print("Fit time: {:.2f} sec".format(end_time-start_time))

start_time = time.time()
print("Evaluating with shrink value = {} and topK = {}".format(sh, k))
result = evaluate_algorithm(URM_test, recommender, userList, at=10)
end_time = time.time()
print("Evaluation time: {:.2f} minutes".format((end_time-start_time)/60))


# evaluation done on half the user pool to speed the things up!
length = len(userList)
half_users = userList[:int(length/2)]

shrink_values = [0, 10, 50, 100, 200]
k_values = [500, 750, 1000]

shrink_results = []
k_results = []

START = time.time()

for tk in k_values:
    print("####################################")
    print("Fitting . . .")
    recommender.fit(shrink=10, topK=tk)
    start_time = time.time()
    print("Evaluating with K value = ", tk)
    result = evaluate_algorithm(URM_test, recommender, half_users, at=10)
    end_time = time.time()
    print("Evaluation time: {:.2f} minutes".format((end_time-start_time)/60))
    k_results.append(result["MAP"])

best_k = k_values[k_results.index(max(k_results))]

for sh in shrink_values:
    print("####################################")
    print("Fitting . . .")
    recommender.fit(shrink=sh, topK=best_k)
    start_time = time.time()
    print("Evaluating with shrink value = ", sh)
    result = evaluate_algorithm(URM_test, recommender, half_users, at=10)
    end_time = time.time()
    print("Evaluation time: {:.2f} minutes".format((end_time-start_time)/60))
    shrink_results.append(result["MAP"])


best_sh = max(shrink_results)


END = time.time()

best_k = k_values[k_results.index(max(k_results))]
best_sh = shrink_values[shrink_results.index((max(shrink_results)))]

total_time = (END - START)/60

print("Total time for parameter tuning is {:.2f} minutes".format(total_time))
print("The best tuning is: \n Shrink = {}\nK = {}".format(best_sh, best_k))

'''

