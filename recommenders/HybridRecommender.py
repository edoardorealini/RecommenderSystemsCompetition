import numpy as np
import scipy.sparse as sps

from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender


class HybridRecommender(SimilarityMatrixRecommender):
    """ ItemKNNSimilarityHybridRecommender
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNSimilarityHybridRecommender"


    def __init__(self, URM_train, Similarity_1, Similarity_2, Similarity_3, sparse_weights=True):
        self.URM = URM_train
        super(HybridRecommender, self).__init__()

        if Similarity_1.shape != Similarity_2.shape or Similarity_3.shape != Similarity_2.shape or Similarity_3.shape != Similarity_1.shape:
            raise ValueError("ItemKNNSimilarityHybridRecommender: similarities have different sizes, S1 is {}, S2 is {}, S3 is {}".format(
                Similarity_1.shape, Similarity_2.shape, Similarity_3.shape
            ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')
        self.Similarity_3 = check_matrix(Similarity_3.copy(), 'csr')

        self.URM_train = check_matrix(URM_train.copy(), 'csr')

        self.sparse_weights = sparse_weights


    def fit(self, topK=10, alpha = 0.5, beta = 0.3, gamma = 0.2):

        self.topK = topK
        self.alpha = alpha
        self.beta = beta
        self.gamma =gamma

        W = self.Similarity_1*self.alpha + self.Similarity_2*self.beta + self.Similarity_3*self.gamma

        if self.sparse_weights:
            self.W_sparse = similarityMatrixTopK(W,
                                                 # forceSparseOutput=True,
                                                 k=self.topK)
        else:
            self.W = similarityMatrixTopK(W,
                                          # forceSparseOutput=False,
                                          k=self.topK)

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

    def save_model(self, name, path = "C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/models/Hybrid"):
        print("[HybridRecommender] Saving model on file " + path + "/" + name + ".npz")
        sps.save_npz(path + "/" + name + ".npz", self.W_sparse, compressed=True)

    def load_model(self, name, path = "C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/models/Hybrid"):
        print("[HybridRecommender] Loading model from file " + path + "/" + name + ".npz")
        self.W_sparse = sps.load_npz(path + "/" + name + ".npz")
        self.W_sparse = self.W_sparse.tocsr()