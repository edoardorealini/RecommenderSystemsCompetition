import numpy as np
import scipy.sparse as sps

from sklearn.preprocessing import normalize
from Base.Recommender_utils import check_matrix, similarityMatrixTopK

from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender
import time, sys
from Notebooks_utils.evaluation_function import evaluate_algorithm, evaluate_algorithm_coldUsers, evaluate_algorithm_original


class RP3betaRecommender(BaseSimilarityMatrixRecommender):
    """ RP3beta recommender """

    RECOMMENDER_NAME = "RP3betaRecommender"

    def __init__(self, URM_train):
        super(RP3betaRecommender, self).__init__(URM_train)
        self.URM = URM_train

    def __str__(self):
        return "RP3beta(alpha={}, beta={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(
            self.alpha,
            self.beta, self.min_rating, self.topK,
            self.implicit, self.normalize_similarity)

    def fit(self, alpha=0.5, beta=0.2, min_rating=0, topK=10, implicit=True, normalize_similarity=False):
        self.alpha = alpha
        self.beta = beta
        self.min_rating = min_rating
        self.topK = topK
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity

        print("[RP3beta]: Fitting with params:", "(alpha={}, beta={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(
            self.alpha,
            self.beta, self.min_rating, self.topK,
            self.implicit, self.normalize_similarity))


        # if X.dtype != np.float32:
        #     print("RP3beta fit: For memory usage reasons, we suggest to use np.float32 as dtype for the dataset")

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)

        # Pui is the row-normalized urm
        Pui = normalize(self.URM_train, norm='l1', axis=1)

        # Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM_train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)

        # Taking the degree of each item to penalize top popular
        # Some rows might be zero, make sure their degree remains zero
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()

        degree = np.zeros(self.URM_train.shape[1])

        nonZeroMask = X_bool_sum != 0.0

        degree[nonZeroMask] = np.power(X_bool_sum[nonZeroMask], -self.beta)

        # ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del (X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = np.multiply(similarity_block[row_in_block, :], degree)
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

            if time.time() - start_time_printBatch > 60:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
                    current_block_start_row,
                    100.0 * float(current_block_start_row) / Pui.shape[1],
                    (time.time() - start_time) / 60,
                    float(current_block_start_row) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(Pui.shape[1], Pui.shape[1]))

        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)

        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)

        self.W_sparse = check_matrix(self.W_sparse, format='csr')

    def compute_score(self, user_id, exclude_seen=True):
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        return scores

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

    def save_model(self, name, path = "C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/models/RP3beta"):
        print("[RP3beta] Saving model on file " + path + "/" + name + ".npz")
        sps.save_npz(path + "/" + name + ".npz", self.W_sparse, compressed=True)

    def load_model(self, name, path = "C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/models/RP3beta"):
        print("[RP3beta] Loading model from file " + path + "/" + name + ".npz")
        self.W_sparse = sps.load_npz(path + "/" + name + ".npz")
        print("[RP3beta] Model loaded correctly")
        self.W_sparse = self.W_sparse.tocsr()

    def get_model(self):
        return self.W_sparse
