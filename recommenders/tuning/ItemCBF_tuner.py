from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

from DataUtils.dataLoader import *
from Base.Evaluation.Evaluator import EvaluatorHoldout

URM_train, URM_test = load_data_split(0)
ICM_all = load_ICM()

recommender = ItemKNNCBFRecommender(URM_train, ICM_all)
evaluator = EvaluatorHoldout(URM_test, [10])

recommender.fit(topK=5, shrink=112, normalize=True, similarity="cosine")
recommender.save_model(name="gino")

dict, string = evaluator.evaluateRecommender(recommender)
print("Evaluation result MAP: {}".format(dict[10]["MAP"]))
