from NotebookLibraries.Notebooks_utils.evaluation_function import evaluate_algorithm_coldUsers
from NotebookLibraries.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import scipy.sparse as sps
import numpy as np
from DataUtils.ParserURM import ParserURM
from DataUtils.ouputGenerator import *
from recommenders import TopPopRecommender as tp


class ItemCFKNNRecommender(object):

    def __init__(self, URM):
        self.URM = URM


    def fit(self, topK=50, shrink=100, normalize=True, similarity="cosine"):

        similarity_object = Compute_Similarity_Python(self.URM, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity = similarity)
        print("[ItemCFKNNRecommender] Fitting . . .")
        self.W_sparse = similarity_object.compute_similarity()


    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]


    def filter_seen(self, user_id, scores):

        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id +1]

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


parser = ParserURM()
URM_path = "data/competition/data_train.csv"
parser.generateURMfromFile(URM_path)

userList = parser.getUserList_unique()

recommender_CF = ItemCFKNNRecommender(URM_all)
recommender_TopPop = tp.TopPopRecommender()

start_time = time.time()
print("###  Training . . .   ###")
recommender_CF.fit(shrink=28, topK=10, similarity="jaccard")
end_time = time.time()
print("Fit time for CF: {:.2f} sec".format(end_time-start_time))

start_time = time.time()
recommender_TopPop.fit(URM_all)
end_time = time.time()
print("Fit time for TopPop: {:.2f} sec".format(end_time-start_time))

# NB: generare output solo sugli utenti che non sono cold !
# per i cold users Usare il top popular trainato su tutta la matrice URM (Senza split)

#create_output_coldUsers(name="ItemCFKNN_consideringCold_03-12", firstRecommender=recommender_CF, coldRecommender=recommender_TopPop)
create_output_coldUsers_Age(name="ItemCFKNN_consideringCold+AGE_07-12_sh28", firstRecommender=recommender_CF)

'''

start_time = time.time()
evaluate_algorithm_coldUsers(URM_test, recommender_CF, recommender_TopPop, at=10)
end_time = time.time()

print("\nEvaluation time: {:.2f} mins".format((end_time-start_time)/60))

length = len(userList)
half_users = userList[:int(length/2)]

# evaluation done on half the user pool to speed the things up!

shrink_values = [0, 10, 50, 100]
k_values = [100, 250, 500, 1000]

shrink_results = []
k_results = []

START = time.time()

for sh in shrink_values:
    print("####################################")
    print("Fitting . . .")
    recommender.fit(shrink=sh)
    start_time = time.time()
    print("Evaluating with shrink value = ", sh)
    result = evaluate_algorithm(URM_test, recommender, half_users, at=10)
    end_time = time.time()
    print("Evaluation time: {:.2f} minutes".format((end_time-start_time)/60))
    shrink_results.append(result["MAP"])


# best_sh = max(shrink_results)
best_sh = 50

for tk in k_values:
    print("####################################")
    print("Fitting . . .")
    recommender.fit(shrink=best_sh, topK=tk)
    start_time = time.time()
    print("Evaluating with K value = ", tk)
    result = evaluate_algorithm(URM_test, recommender, half_users, at=10)
    end_time = time.time()
    print("Evaluation time: {:.2f} minutes".format((end_time-start_time)/60))
    k_results.append(result["MAP"])

END = time.time()

best_k = max(k_results)
total_time = (END - START)/60

print("Total time for parameter tuning is {:.2f} minutes".format(total_time))
print("The best tuning is: \n Shrink = {}\nK = {}".format(best_sh, best_k))
'''

