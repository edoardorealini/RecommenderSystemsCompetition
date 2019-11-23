import time


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
    output = open("output/" + name + ".csv", 'w')
    output.write("user_id,item_list\n")
    print("[OutputGenerator] starting to generate recommendations")
    start_time = time.time()

    for user in getUserList_forOutput():
        recommended_items = recommender.recommend(user, at=10)
        output.write(list_to_output(user, recommended_items))

    end_time = time.time()
    print("[OutputGenerator] output correctly written on file " + name + ".csv in {:.2f} mins".format((end_time - start_time)/60))