from Base.Recommender_utils import check_matrix
import numpy as np


class LinearHybridRecommender():
    """ LinearHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "LinearHybridRecommender"

    def __init__(self, URM_train, Recommender_1, Recommender_2, Recommender_3):
        super(LinearHybridRecommender, self).__init__()
        self.URM_train = check_matrix(URM_train.copy(), 'csr')

        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        self.Recommender_3 = Recommender_3

        # i think this following line is pretty useless.
        self.compute_item_score = self.compute_scores_hybrid

    def fit(self, alpha=1.0, beta=1.0, gamma=1.0, retrain_all_algorithms=False):
        if retrain_all_algorithms:
            self.Recommender_1.fit()
            self.Recommender_2.fit()
            self.Recommender_3.fit()

        self.alpha = alpha
        self.beta = beta
        self.gamma =gamma

    def compute_scores_hybrid(self, user_id):
        item_weights_1 = self.Recommender_1.compute_score(user_id)
        item_weights_2 = self.Recommender_2.compute_score(user_id)
        item_weights_3 = self.Recommender_3.compute_score(user_id)

        item_weights = item_weights_1 * self.alpha + \
                       item_weights_2 * self.beta + \
                       item_weights_3 * self.gamma

        return item_weights

    def recommend(self, user_id, at=10, exclude_seen=True):
        scores = self.compute_scores_hybrid(user_id)

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):

        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores
