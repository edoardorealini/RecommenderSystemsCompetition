from DataUtils.dataLoader import *
from recommenders.old_Algorithms.CBFRecommender import ItemCBFKNNRecommender

URM_all, URM_train, URM_test = load_all_data()
ICM_all = load_ICM()

recommender = ItemCBFKNNRecommender(URM=URM_train, ICM=ICM_all)
recommender.fit()
recommender.save_model(name="test7.3_URM_train")


'''

shrink_values = [1, 5, 10, 30, 60, 100]
shrink_results = []

START = time.time()

for sh in shrink_values:
    print("####################################")
    print("Fitting . . .")
    recommender.fit(shrink=sh, topK=10)
    start_time = time.time()
    print("Evaluating with shrink value = ", sh)
    result = evaluate_algorithm_original(URM_test, recommender, at=10)
    end_time = time.time()
    print("Evaluation time: {:.2f} minutes".format((end_time-start_time)/60))
    shrink_results.append(result["MAP"])

END = time.time()

total_time = (END - START)/60

print("Total time for parameter tuning is {:.2f} minutes".format(total_time))



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

    return list(userList)

def list_to_output(user, list_of_elements):
    string = str(user) + ","

    for item in list_of_elements:
        string = string + str(item) + " "

    return string + "\n"

user_list = getUserList_forOutput()

def create_output(name, recommender):
    output = open("output/" + name + ".csv", 'w')
    output.write("user_id,item_list\n")
    for user in user_list:
        recommended_items = recommender.recommend(user, at=10, exclude_seen=False)
        output.write(list_to_output(user, recommended_items))

'''