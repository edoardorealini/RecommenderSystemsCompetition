import numpy as np
import scipy.sparse as sps
from Notebooks_utils.data_splitter import train_test_holdout
from Notebooks_utils.evaluation_function import evaluate_algorithm

URM_all = sps.load_npz('data/competition/sparse_URM.npz')
print("URM correctly loaded from file: data/competition/sparse_URM.npz")

URM_test = sps.load_npz('data/competition/sparse_URM_test.npz')
print("URM_test correctly loaded from file: data/competition/sparse_URM_test.npz")

URM_train = sps.load_npz('data/competition/sparse_URM_train.npz')
print("URM_train correctly loaded from file: data/competition/sparse_URM_train.npz")


# Random recommender
class RandomRecommender(object):
    def fit(self, URM_train):
        self.numItems = URM_train.shape[0]

    def recommend(self, user_id, at=5):
        recommended_items = np.random.choice(self.numItems, at)

        return recommended_items


# Splitting data, done, saved in files (load from there from now on)
# URM_train, URM_test = train_test_holdout(URM_all, train_perc = 0.8)
# sps.save_npz('data/competition/sparse_URM_train.npz', URM_train, compressed=True)
# sps.save_npz('data/competition/sparse_URM_test.npz', URM_test, compressed=True)

user_id = np.random.randint(low=0, high=10000)

random_recommender = RandomRecommender()

random_recommender.fit(URM_train)
recommended_items = random_recommender.recommend(user_id, at=10)
print("Items recommended for user", user_id, recommended_items)

evaluate_algorithm(URM_test, random_recommender)