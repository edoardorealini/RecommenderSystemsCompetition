from recommenders.ItemCFKNNRecommender import ItemCFKNNRecommender
from DataUtils.datasetSplitter import datasetSplitter

# shrink tuning for the ItemKNNCFRecommender
# This file can then be generalized to tune the shrink value of differenti kind of recommenders

URM_path = "./data/competition/data_train.csv"

splitter = datasetSplitter()
splitter.loadURMdata(URM_path)

URM_train, URM_test = splitter.splitDataBetter()