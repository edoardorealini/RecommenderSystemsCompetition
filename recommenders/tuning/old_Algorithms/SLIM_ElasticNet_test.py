import scipy.sparse as sps
from Notebooks_utils.evaluation_function import evaluate_algorithm_original
from recommenders.old_Algorithms.SLIM_ElasticNet import SLIMElasticNetRecommender

evaluate_algorithm = False
load_model = False

URM_all = sps.load_npz('C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/sparse_URM.npz')
print("URM correctly loaded from file: data/competition/sparse_URM.npz")
URM_all = URM_all.tocsr()

URM_test = sps.load_npz('C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/URM_test.npz')
print("URM_test correctly loaded from file: data/competition/URM_test.npz")
URM_test = URM_test.tocsr()

URM_train = sps.load_npz('C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/URM_train.npz')
print("URM_train correctly loaded from file: data/competition/URM_train.npz")
URM_train = URM_train.tocsr()

print(type(URM_train))

recommender = SLIMElasticNetRecommender(URM_all)
if not load_model:
    print("[SLIMElasticNet_test]: Fitting . . .")
    recommender.fit()
    recommender.save_model(name="model_URM_all")

if load_model:
    recommender.load_model(name="test3_URM_train")

if evaluate_algorithm:
    evaluate_algorithm_original(URM_test, recommender, at=10)
