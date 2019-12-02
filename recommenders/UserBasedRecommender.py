'''

    This is a User Based Recommender made for the evaluation of cold users

'''
import numpy as np
from NotebookLibraries.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python


class UserBasedRecommender():

    def __init__(self, UCM, CFRecommender):
        self.UCM = UCM.tocsr()
        self.CFRecommender = CFRecommender
        self.W_sparse = UCM

    def fit(self):
        def fit(self, topK=10, shrink=50, normalize=True, similarity="jaccard"):
            similarity_object = Compute_Similarity_Python(self.UCM.T, shrink=shrink,
                                                          topK=topK, normalize=normalize,
                                                          similarity=similarity)

            self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=10):
        # compute the scores using the dot product
        #Idea: searching in sim matrix the most similar user to a cold one and recommending
        #   with a CF with the known data (user for which i know things)

        mostSimilarUserID = 0

        similarUsers = self.W_sparse[user_id]
        mostSimilarUserID = similarUsers.argmax()

        if user_id == 20:
            print(type(mostSimilarUserID))
            print(mostSimilarUserID)

        scores = self.CFRecommender.recommend(mostSimilarUserID, at=10)

        return scores
