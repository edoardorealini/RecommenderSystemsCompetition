from NotebookLibraries.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import scipy.sparse as sps
from NotebookLibraries.Notebooks_utils.evaluation_function import evaluate_algorithm

URM_train = sps.load_npz("./data/competition/URM_train.npz")
URM_test = sps.load_npz("./data/competition/URM_test.npz")

recommender = SLIM_BPR_Cython(URM_train)
recommender.fit(epochs=5)

evaluate_algorithm(URM_test, recommender, at=10)

