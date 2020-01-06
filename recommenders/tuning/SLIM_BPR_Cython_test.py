from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

from DataUtils.dataLoader import *
from Base.Evaluation.Evaluator import EvaluatorHoldout

URM_all = load_URM_all()
URM_train, URM_test = load_data_split(0)

recommender = SLIM_BPR_Cython(URM_train, verbose=False, recompile_cython=False)
evaluator = EvaluatorHoldout(URM_test, [10])

recommender.fit(epochs=1000)
recommender.save_model(name="new_split_fun")

dict, string = evaluator.evaluateRecommender(recommender)
print("Evaluation result MAP: {}".format(dict[10]["MAP"]))