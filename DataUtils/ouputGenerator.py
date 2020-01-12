import time
from DataUtils.datasetSplitter import *
import os
from tqdm import tqdm
from DataUtils.UserSplitterByAge import *
from DataUtils.dataLoader import *
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

def getUserList_forOutput():
    userList_file = open("C:/Users/Utente/Desktop/RecSys-Competition-2019/data/competition/alg_sample_submission.csv", 'r')

    def rowSplit(rowString):
        split = rowString.split(",")

        split[0] = int(split[0])
        result = split[0]
        return result

    userList_file.seek(0)
    userList = []

    for line in userList_file:
        userList.append(rowSplit(line))

    print("[OutputGenerator] got user list for output creation")
    return list(userList)


def list_to_output(user, list_of_elements):
    string = str(user) + ","

    for item in list_of_elements:
        string = string + str(item) + " "

    return string + "\n"


def create_output(name, recommender):
    abspath = os.path.abspath("output/"+name+".csv")
    output = open(abspath, 'w')
    output.write("user_id,item_list\n")
    print("[OutputGenerator] starting to generate recommendations")
    start_time = time.time()

    for user in tqdm(getUserList_forOutput()):
        if user % 5000 == 0:
            print("Recommending to user ", user)

        recommended_items = recommender.recommend(user, at=10)
        output.write(list_to_output(user, recommended_items))

    end_time = time.time()
    print("[OutputGenerator] output correctly written on file " + name + ".csv in {:.2f} mins".format((end_time - start_time)/60))


def create_output_coldUsers(name, firstRecommender, coldRecommender):
    coldUserList = getColdUsers()

    abspath = os.path.abspath("output/" + name + ".csv")
    output = open(abspath, 'w')

    output.write("user_id,item_list\n")

    print("[OutputGenerator] starting to generate recommendations")

    start_time = time.time()

    for user in tqdm(getUserList_forOutput()):
        #if user % 5000 == 0:
        #    print("Recommending to user ", user)

        if user in coldUserList:
            recommended_items = coldRecommender.recommend(user, at=10)

        else:
            recommended_items = firstRecommender.recommend(user, at=10)

        output.write(list_to_output(user, recommended_items))

    end_time = time.time()

    print("[OutputGenerator] output correctly written on file " + name + ".csv in {:.2f} mins".format((end_time - start_time)/60))


def getUserAge(userId):
    age = 0
    usersByAge = getAllUsersByAge()

    for age in range(1,11):
        if userId in usersByAge[age]:
            return age

    # returns 0 if there is no info on the user's age!
    return 0


def recommendTopPopOnAge(age):
    std_topPop = [17955, 8638, 5113, 10227, 4657, 197, 8982, 10466, 3922, 4361]
    if age == 0:
        return std_topPop

    list_of_recommends = []

    file = open("C:/Users/Utente/Desktop/RecSys-Competition-2019/data/competition/recommends_by_age.txt", "r")

    for line in file:
        line = line.replace("\n", "")
        split = line.split(",")

        if int(split[0]) == age:
            list_of_recommends = split[1].split(" ")

    list_of_recommends = list_of_recommends[1:]
    return list_of_recommends


def create_output_coldUsers_Age(output_name, recommender, use_cbf_on_tails=False, cbf_tail_length=0):
    coldUserList = getColdUsers()

    URM_all = load_URM_all()
    ICM_all = load_ICM()
    CBFRecommender = ItemKNNCBFRecommender(URM_train=URM_all, ICM_train=ICM_all)
    CBFRecommender.load_model(name="model_URM_all")
    CBFRecommends = []

    # abspath = os.path.abspath("output/" + output_name + ".csv")
    abspath = "C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/output/" + output_name + ".csv"
    output = open(abspath, 'w')

    output.write("user_id,item_list\n")

    print("[OutputGenerator] starting to generate recommendations")
    print("[OutputGenerator] using CF on normal users and TopPop with age info on Cold Users!")

    if use_cbf_on_tails:
        print("[outputGenerator] Using CBF on tails of recommends to introduce unseen items with higher probability")

    start_time = time.time()

    for user in tqdm(getUserList_forOutput()):
        if user in coldUserList:
            recommended_items = recommendTopPopOnAge(getUserAge(user))

        else:
            if use_cbf_on_tails:
                CBFRecommends = CBFRecommender.recommend(user)[0:20]
                recommended_items = recommender.recommend(user)[0:10-cbf_tail_length]
                recommended_items = list(recommended_items)

                for iteration in range(cbf_tail_length):
                    for item in CBFRecommends:
                        if item not in recommended_items:
                            recommended_items.append(item)
                            CBFRecommends.remove(item)
                            break

            else:
                recommended_items = recommender.recommend(user)[0:10]

        output.write(list_to_output(user, recommended_items))

    end_time = time.time()
    print("[OutputGenerator] output generated considering age when recommending topPop!")
    print("[OutputGenerator] output correctly written on file " + output_name + ".csv in {:.2f} mins".format((end_time - start_time) / 60))

# input: less than interactions
# This values is needed to select only the users that have less than n_interactions- interactions
def get_pseudoCold_users(n_interactions=2):
    pseudo_cold = []

    #TODO

    return pseudo_cold


'''
With this following function we want to distinguish between 3 different categories of users:
    - Cold Users:           Using TOP POP clustered on age
    - Pseudo Cold Users :   Using hybrid with higher weight on CBF recommender
    - Normal Users:         Using the top tuned hybrid recommender
    
    A user is considered Pseudo Cold User if he has only 1 or 2 interactions in the whole dataset. 
'''

def create_output_superColdMng(output_name, hybrid_CF, hybrid_CBF):
    coldUserList = getColdUsers()

    path = "C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/output/" + output_name + ".csv"
    output = open(path, 'w')

    output.write("user_id,item_list\n")

    print("[OutputGenerator] starting to generate recommendations")
    print("[OutputGenerator] \nUsing: Hybrid CF on normal users, \nHybrid with higher weight on CBF for Pseudo Cold Users (less than 2 interations) \nTopPop with age info on Cold Users!")

    start_time = time.time()

    for user in tqdm(getUserList_forOutput()):
        if user in coldUserList:
            recommended_items = recommendTopPopOnAge(getUserAge(user))

        else:
            recommended_items = hybrid_CF.recommend(user)[0:10]

        output.write(list_to_output(user, recommended_items))

    end_time = time.time()
    print("[OutputGenerator] output generated considering age when recommending topPop!")
    print("[OutputGenerator] output correctly written on file " + output_name + ".csv in {:.2f} mins".format(
        (end_time - start_time) / 60))
