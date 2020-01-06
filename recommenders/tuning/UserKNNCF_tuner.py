from KNN.UserKNNCFRecommender import UserKNNCFRecommender

from DataUtils.dataLoader import *
from Base.Evaluation.Evaluator import EvaluatorHoldout

URM_train, URM_test = load_data_split(0)

recommender = UserKNNCFRecommender(URM_train)
evaluator = EvaluatorHoldout(URM_test, [10])

recommender.fit(topK=50, shrink=0.05, normalize=True, similarity="cosine")
recommender.save_model(name="test_with_new_params")

dict, string = evaluator.evaluateRecommender(recommender)
print("Evaluation result MAP: {}".format(dict[10]["MAP"]))