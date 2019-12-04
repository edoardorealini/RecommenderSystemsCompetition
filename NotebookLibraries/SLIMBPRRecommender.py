import time
import scipy.sparse as sps
from NotebookLibraries.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import NotebookLibraries
from NotebookLibraries.Notebooks_utils.evaluation_function import evaluate_algorithm_coldUsers
from NotebookLibraries.Notebooks_utils.evaluation_function import evaluate_algorithm
from NotebookLibraries.Base.Evaluation.Evaluator import SequentialEvaluator

URM_all = sps.load_npz('data/competition/sparse_URM.npz')
print("URM correctly loaded from file: data/competition/sparse_URM.npz")
URM_all = URM_all.tocsr()

URM_test = sps.load_npz('data/competition/URM_test.npz')
print("URM_test correctly loaded from file: data/competition/URM_test.npz")
URM_test = URM_test.tocsr()

URM_train = sps.load_npz('data/competition/URM_train.npz')
print("URM_train correctly loaded from file: data/competition/URM_train.npz")
URM_train = URM_train.tocsr()


recommender = SLIM_BPR_Cython(URM_train, recompile_cython=False)
recommender.fit(epochs=20, batch_size=1, sgd_mode='sgd', learning_rate=1e-4)

start_time = time.time()
evaluate_algorithm(URM_test, recommender,  at=10)
end_time = time.time()