import time
from DataUtils.datasetSplitter import *
import os
from tqdm import tqdm
from DataUtils.UserSplitterByAge import *


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

    file = open("./data/competition/recommends_by_age.txt", "r")

    for line in file:
        line = line.replace("\n", "")
        split = line.split(",")

        if int(split[0]) == age:
            list_of_recommends = split[1].split(" ")

    list_of_recommends = list_of_recommends[1:]
    return list_of_recommends


def create_output_coldUsers_Age(output_name, recommender):
    coldUserList = getColdUsers()

    abspath = os.path.abspath("output/" + output_name + ".csv")
    output = open(abspath, 'w')

    output.write("user_id,item_list\n")

    print("[OutputGenerator] starting to generate recommendations")
    print("[OutputGenerator] using CF on normal users and TopPop with age info on Cold Users!")

    start_time = time.time()

    for user in tqdm(getUserList_forOutput()):
        if user in coldUserList:
            recommended_items = recommendTopPopOnAge(getUserAge(user))

        else:
            recommended_items = recommender.recommend(user, at=10)

        output.write(list_to_output(user, recommended_items))

    end_time = time.time()
    print("[OutputGenerator] output generated considering age when recommending topPop!")
    print("[OutputGenerator] output correctly written on file " + output_name + ".csv in {:.2f} mins".format((end_time - start_time) / 60))