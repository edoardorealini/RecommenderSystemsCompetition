#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender_utils import check_matrix
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np
import scipy.sparse as sps

from Base.Similarity.Compute_Similarity import Compute_Similarity


class UserKNNCBFRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCBFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, URM_train, UCM_train, verbose=True):
        super(UserKNNCBFRecommender, self).__init__(URM_train, verbose=verbose)

        self.UCM_train = UCM_train

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting="none",
            **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError(
                "Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                    self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        if feature_weighting == "BM25":
            self.UCM_train = self.UCM_train.astype(np.float32)
            self.UCM_train = okapi_BM_25(self.UCM_train)

        elif feature_weighting == "TF-IDF":
            self.UCM_train = self.UCM_train.astype(np.float32)
            self.UCM_train = TF_IDF(self.UCM_train)

        similarity = Compute_Similarity(self.UCM_train.T, shrink=shrink, topK=topK, normalize=normalize,
                                        similarity=similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')

    def save_model(self, name, path="C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/models/UserCBF"):
        print("[UserCBF] Saving model on file " + path + "/" + name + ".npz")
        sps.save_npz(path + "/" + name + ".npz", self.W_sparse, compressed=True)

    def load_model(self, name, path="C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/models/UserCBF"):
        print("[UserCBF] Loading model from file " + path + "/" + name + ".npz")
        self.W_sparse = sps.load_npz(path + "/" + name + ".npz")
        print("[UserCBF] Model loaded correctly")
        self.W_sparse = self.W_sparse.tocsr()

    def get_model(self):
        return self.W_sparse