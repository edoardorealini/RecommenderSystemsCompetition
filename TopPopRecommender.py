import numpy as np
import scipy.sparse as sps
from Notebooks_utils.data_splitter import train_test_holdout
from Notebooks_utils.evaluation_function import evaluate_algorithm
from ParserURM import ParserURM


'''
URM_all = sps.load_npz('data/competition/sparse_URM.npz')
print("URM correctly loaded from file: data/competition/sparse_URM.npz")

URM_test = sps.load_npz('data/competition/sparse_URM_test.npz')
print("URM_test correctly loaded from file: data/competition/sparse_URM_test.npz")
URM_test.tocsr()

URM_train = sps.load_npz('data/competition/sparse_URM_train.npz')
print("URM_train correctly loaded from file: data/competition/sparse_URM_train.npz")
URM_train.tocsr()

def getUserList():
    userList_file = open("data/competition/alg_sample_submission.csv", 'r')

    def rowSplit(rowString):
        split = rowString.split(",")

        split[0] = int(split[0])
        result = split[0]
        return result

    userList_file.seek(0)
    userList = []

    for line in userList_file:
        userList.append(rowSplit(line))

    return list(userList)

userList_unique = getUserList()
#userList_unique = list(set(userList))

print(userList_unique)
'''


class TopPopRecommender(object):

    def fit(self, URM_train):
        itemPopularity = (URM_train > 0).sum(axis=0)
        itemPopularity = np.array(itemPopularity).squeeze()

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popularItems = np.argsort(itemPopularity)
        self.popularItems = np.flip(self.popularItems, axis=0)

    def recommend(self, user_id, at=5):
        recommended_items = self.popularItems[0:at]

        return recommended_items



# Data already splitted (spero correttamente)
'''

user_id = np.random.randint(low=0, high=10000)

topPop_recommender = TopPopRecommender()

topPop_recommender.fit(URM_all)
# fit on the entire dataset, in this way i find the top items without excluding some: OVERFITTING BESTIA DE DIO
recommended_items = topPop_recommender.recommend(user_id, at=10)
print("Items recommended for user", user_id, recommended_items)

parser = ParserURM()
URM_path = "data/competition/data_train.csv"
parser.generateURMfromFile(URM_path)

userList = parser.getUserList_unique()

evaluate_algorithm(URM_test, topPop_recommender, userList)

def list_to_output(user, list_of_elements):
    string = str(user) + ","

    for item in list_of_elements:
        string = string + str(item) + " "

    return string + "\n"

#testing one print
recommended_items = topPop_recommender.recommend(100, at=10)
print(list_to_output(100, recommended_items))

output = open("output/top_popular.csv", 'w')
output.write("user_id,item_list\n")
for user in userList_unique:
    recommended_items = topPop_recommender.recommend(user, at=10)
    output.write(list_to_output(user, recommended_items)) 


'''