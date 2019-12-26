from Notebooks_utils import evaluate_algorithm
from recommenders.ItemCFKNNRecommender import ItemCFKNNRecommender
from DataUtils.datasetSplitter import datasetSplitter

# shrink tuning for the ItemKNNCFRecommender
# This file can then be generalized to tune the shrink value of differenti kind of recommenders

URM_path = "./data/competition/data_train.csv"

splitter = datasetSplitter()
splitter.loadURMdata(URM_path)
URM_train, URM_test = splitter.splitDataBetter()

# now that we can split data in a decent timing, we can start the evaluation of the best shrink parameter, keeping k = 10

shrink_values = []
results = []

for shrink in range(15, 26):
    shrink_values.append(shrink)

print(shrink_values)
# Testing with a single single split value!
for sh in shrink_values:
    recommender = ItemCFKNNRecommender(URM_train)
    recommender.fit(topK=10, shrink=sh, normalize=True, similarity="jaccard")
    result = evaluate_algorithm(URM_test, recommender, at=10)
    results.append(result)

values = range(15, 26)
i = 0
for r in results:
    print(values[i], r)
    i += 1