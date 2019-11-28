from DataUtils.ParserURM import ParserURM
from DataUtils.ouputGenerator import *

URM_path = "data/competition/data_train.csv"

userList_complete = []
userList_output = []

parser = ParserURM()
parser.generateURMfromFile(URM_path)

# in userList_complete we find the users that have at least one interaction with an item in the URM
userList_complete = parser.getUserList_unique()
# in the userList_output we can find the users for which we have to make recommendations in output
userList_output = getUserList_forOutput()

print("User list expected in output: ", userList_output[:10])
print("User list found in the original URM: ", userList_complete[:10])

output_file = open("output/userlist_output.txt", 'w')
for user in userList_output:
    output_file.write(str(user))
    output_file.write("\n")

complete_file = open("output/userlist_complete.txt", 'w')
for user in userList_complete:
    complete_file.write(str(user))
    complete_file.write("\n")


# check which users are in the output list but not in the complete one
cold_users = []

for user in userList_output:
    if user % 5000 == 0:
        print("Evaluating userID number ", user, " of ", len(userList_output))

    try:
        userList_complete.index(user)
    except:
        cold_users.append(user)


print("Cold user are: ", cold_users[:10])
print("Total number of cold users: ", len(cold_users))

print("Writing on file ", "output/userlist_cold.txt", " the list of cold_users")
cold_file = open("output/userlist_cold.txt", 'w')
for user in cold_users:
    cold_file.write(str(user))
    cold_file.write("\n")