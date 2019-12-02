from NotebookLibraries.Notebooks_utils.evaluation_function import evaluate_algorithm_coldUsers
from NotebookLibraries.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import scipy.sparse as sps
import numpy as np
from DataUtils.ParserURM import ParserURM
from DataUtils.ouputGenerator import *
from recommenders import TopPopRecommender as tp

UCM_path = "./data/competition/data_UCM_age.csv"
UCM_file = open(UCM_path, "r")

age_tuples = []

for row in UCM_file:
    age_tuples.append(rowSplit(row))

age_users, age_values, age_ones = createLists(age_tuples)

age_values_unique = list(set(age_values))

one = []
two = []
three = []
four = []
five = []
six = []
seven = []
eight = []
nine = []
ten = []

for tuple in age_tuples:
    age = tuple[1]
    user = tuple[0]
    if age == 1:
        one.append(user)
    if age == 2:
        two.append(user)
    if age == 3:
        three.append(user)
    if age == 4:
        four.append(user)
    if age == 5:
        five.append(user)
    if age == 6:
        six.append(user)
    if age == 7:
        seven.append(user)
    if age == 8:
        eight.append(user)
    if age == 9:
        nine.append(user)
    if age == 10:
        ten.append(user)

users_by_age = []
users_by_age.append([1,1])
users_by_age.append(one)
users_by_age.append(two)
users_by_age.append(three)
users_by_age.append(four)
users_by_age.append(five)
users_by_age.append(six)
users_by_age.append(seven)
users_by_age.append(eight)
users_by_age.append(nine)
users_by_age.append(ten)


recommender = tp.TopPopRecommender()
#In users_by_age wi have in position one the list of users that have 1 as age.
#we want to crate a URM with only such users, train a top pop algorithm, store the data and go again

for age in range(1,11):

    URM_all = sps.load_npz('data/competition/sparse_URM.npz')
    print("URM correctly loaded from file: data/competition/sparse_URM.npz")
    URM_all = URM_all.tocsr()

    for user in users_by_age[age]:
        sps.vstack([URM_all[:user,:], URM_all[user:, :]])
        recommender = tp.TopPopRecommender(URM_all)

        

