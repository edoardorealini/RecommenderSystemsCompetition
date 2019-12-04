## Generating a list of the top popular items by AGE!
#idea: creating a file with:
#   AGE, list of items.

#How to do it:
    # we should train multiple times a top popular by giving to it only the users of a certain age
    # known that age ranges from 1 to 10 including the external values+

    #1 Generate a sorted list of users for each age.

import scipy.sparse as sps
from tqdm import tqdm
from DataUtils.datasetSplitter import rowSplit, createLists

UCM_path = "./data/competition/data_UCM_age.csv"

def splitOnAge(UCM_path):
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
    print(users_by_age[1])

    file = open("./data/competition/users_by_age.txt", 'w')

    for age in tqdm(range(1,11)):
        line = ""
        line = str(age) + ", "

        for user in users_by_age[age]:
            line = line + str(user) + " "

        print("[AgeSplitter]: Writing on file . . .")

        file.write(line + "\n")

    file.close()

# given a line returns a list containing users of that age users.
def readUserLine(line):

    #removing the first 2 chars (indicative of age!)
    returnList = []
    terminator = '\n'
    split = line.split(" ")
    split = split[1:]
    #parsing to integers
    for item in split:
        if item != terminator:
            item = int(item)
            returnList.append(item)

    return returnList


# returns a list of lists, containing nothing in pos = 0, the list of users with age 1 in pos = 1 and so on.
# This function works on the data written on file "users_by_age.txt" if not present, it will generate it !
def getAllUsersByAge():

    try:
        file = open("./data/competition/users_by_age.txt", "r")

    except:
        splitOnAge(UCM_path)
        file = open("./data/competition/users_by_age.txt", "r")

    usersByAge = ["empty"]

    for line in file:
        usersByAge.append(readUserLine(line))

    return usersByAge