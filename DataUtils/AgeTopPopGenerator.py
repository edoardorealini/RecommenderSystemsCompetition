##IDEA: creating a URM tailored for the age of the users (keeping only the users of a certain age)
##  then training the TopPop model on that URM and store the restults in a coherent way on a file.

import DataUtils.ParserURM as parser
from DataUtils.UserSplitterByAge import *
import recommenders.TopPopRecommender as tp
import scipy.sparse as sps
from tqdm import tqdm

def createLine(age, listOfRecommends):
    line = ""
    line = line + str(age) + ","

    for rec in listOfRecommends:
        line = line + " " + str(rec)

    return line + "\n"

URM_path = './data/competition/AgeURM/data_train.csv'

parser = parser.ParserURM()
parser.generateURMfromFile(URM_path)
tuples = parser.getTuples()

print(tuples[:10])

recommender = tp.TopPopRecommender()

usersByAge = getAllUsersByAge()
# finalmente riesco a leggere sta merda di dati in lista di liste
# now i have to create the URM personaized for each user list

recommends_file = open("./data/competition/AgeURM/recommends_by_age.txt", "w")

for age in range(1,11):
    #here i have to build the URM on the list of users alone
    agedUsers = usersByAge[age]
    print("[AgeTopPopGen] Loaded users of age: ", age)

    all_users = parser.getUserList()
    all_items = parser.getItemList()
    ratings = parser.getRatingsList()
    print("[AgeTopPopGen] Loaded all users, items and ratings")


    #TOO SLOW implementation!!
    '''
    for i in tqdm(range(0, len(all_users) + 1)):
        if all_users[i] not in agedUsers:
            all_users.remove(all_users[i])
            all_items.remove(all_items[i])
            ratings.remove(ratings[i])
    '''
    print("[AgeTopPopGen] Generating URM")
    URM_Aged = sps.coo_matrix((ratings, (all_users, all_items)))
    URM_Aged = URM_Aged.tocsr()

    print("[AgeTopPopGen] reformatting URM")
    URM_Aged = URM_Aged[agedUsers, :]

    sps.save_npz("./data/competition/AgeURM/URM_age_"+str(age)+".npz", URM_Aged)
    print("[AgeTopPopGen] URM generated and saved on file for further use")

    print("[AgeTopPopGen] Training TopPop model")
    recommender.fit(URM_Aged)

    print("[AgeTopPopGen] Recommending to users of age: ", age)
    recommends = recommender.recommend(user_id=200, at=10)

    print("[AgeTopPopGen] Writing results on fileee")
    recommends_file.write(createLine(age, recommends))

recommends_file.close()
