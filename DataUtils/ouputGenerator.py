import time
from DataUtils.datasetSplitter import *
import os
from tqdm import tqdm

def getUserList_forOutput():
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