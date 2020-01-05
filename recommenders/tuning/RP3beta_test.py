import scipy.sparse as sps
from Notebooks_utils.evaluation_function import evaluate_algorithm_original
from recommenders.RP3betaGraphBased import RP3betaRecommender

URM_all = sps.load_npz('C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/sparse_URM.npz')
print("URM correctly loaded from file: data/competition/sparse_URM.npz")
URM_all = URM_all.tocsr()

URM_test = sps.load_npz('C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/URM_test.npz')
print("URM_test correctly loaded from file: data/competition/URM_test.npz")
URM_test = URM_test.tocsr()

URM_train = sps.load_npz('C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/URM_train.npz')
print("URM_train correctly loaded from file: data/competition/URM_train.npz")
URM_train = URM_train.tocsr()

recommender = RP3betaRecommender(URM_train)
recommender.fit()

evaluate_algorithm_original(URM_test, recommender)
# evaluate_algorithm_coldUsers(URM_test, recommender)

# okay, now that it works let's try to maximize this thing
# Fixing alpha and searching for beta

'''

beta = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
alpha_value = 0.5
results = []

print("Searching for best parameters")
for beta_value in beta:
    start_time = time.time()
    recommender.fit(topK=10, normalize_similarity=False, implicit=True, alpha=alpha_value, beta=beta_value)
    results.append(evaluate_algorithm_original(URM_test, recommender, at=10))
    end_time = time.time()
    print("Total time for this evaluation: ", end_time-start_time, " sec")

beta_value = 0.1
for res in results:
    print("Results obtained with beta: ", beta_value, " and alpha fixed to: 0.5")
    print(res)
    beta_value += 0.1


alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
beta_value = 0.2
results = []

print("Searching for best parameters")
for alpha_value in alpha:
    start_time = time.time()
    recommender.fit(topK=10, normalize_similarity=False, implicit=True, alpha=alpha_value, beta=beta_value)
    results.append(evaluate_algorithm_original(URM_test, recommender, at=10))
    end_time = time.time()
    print("Total time for this evaluation: ", end_time-start_time, " sec")

alpha_value = 0.1
for res in results:
    print("Results obtained with alpha: ", alpha_value, " and beta fixed to: 0.2")
    print(res)
    alpha_value += 0.1
     
'''