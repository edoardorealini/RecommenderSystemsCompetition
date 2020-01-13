from DataUtils.ouputGenerator import *



file = open("C:/Users/Utente/Desktop/RecSys-Competition-2019/DataUtils/data/competition/users/interactions_per_user.txt")

for interactions in range(100):
    counter = 0
    file.seek(0)

    for line in file:
        line.replace("\n", "")
        itr = int(line)
        if itr == interactions:
            counter += 1

    print("There are {} users with {} interactions".format(counter, interactions))


users = get_pseudoCold_users()

print(len(users))

print(users)
