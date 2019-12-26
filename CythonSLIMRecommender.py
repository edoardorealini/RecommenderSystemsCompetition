from SLIM_BPR.SLIM_BPR import SLIM_BPR_Cython
import scipy.sparse as sps
from Base.Evaluation import Evaluator

URM_train = sps.load_npz("./data/competition/URM_train.npz")
URM_test = sps.load_npz("./data/competition/URM_test.npz")

recommender = SLIM_BPR_Cython(URM_train)
recommender.fit(epochs=5)

evaluator = Evaluator(URM_test, cutoff_list=[5])
evaluator.evaluateRecommender(recommender)
