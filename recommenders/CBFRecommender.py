import numpy as np
import scipy.sparse as sps
import time
from NotebookLibraries.Notebooks_utils.evaluation_function import evaluate_algorithm
from NotebookLibraries.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from DataUtils.ParserURM import ParserURM

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

class ItemCBFKNNRecommender(object):

    def __init__(self, URM, ICM):
        self.URM = URM
        self.ICM = ICM

    def fit(self, topK=50, shrink=100, normalize=True, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(self.ICM.T, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=1, exclude_seen=True):
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
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

parser = ParserURM()
URM_path = "data/competition/data_train.csv"
parser.generateURMfromFile(URM_path)

userList = parser.getUserList_unique()

CBFRecommmender = ItemCBFKNNRecommender(URM=URM_train, ICM=ICM_all)
start_time = time.time()
CBFRecommmender.fit(shrink=0.0, topK=100)
end_time = time.time()
print("Fit time: {:.2f} sec".format(end_time-start_time))

start_time = time.time()
evaluate_algorithm(URM_test, CBFRecommmender, userList)
end_time = time.time()
print("Evaluation time: {:.2f} sec".format(end_time-start_time))

def getUserList_forOutput():
    userList_file = open("data/competition/alg_sample_submission.csv", 'r')

    def rowSplit(rowString):
        split = rowString.split(",")

        split[0] = int(split[0])
        result = split[0]
        return result

    userList_file.seek(0)
    userList = []

    for line in userList_file:
        userList.append(rowSplit(line))

    return list(userList)

def list_to_output(user, list_of_elements):
    string = str(user) + ","

    for item in list_of_elements:
        string = string + str(item) + " "

    return string + "\n"

user_list = getUserList_forOutput()

def create_output(name, recommender):
    output = open("output/" + name + ".csv", 'w')
    output.write("user_id,item_list\n")
    for user in user_list:
        recommended_items = recommender.recommend(user, at=10, exclude_seen=False)
        output.write(list_to_output(user, recommended_items))