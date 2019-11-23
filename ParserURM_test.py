from ParserURM import ParserURM

# Some tests of the Class
parser = ParserURM()
URM_path = "data/competition/data_train.csv"
parser.generateURMfromFile(URM_path)

URM = parser.getURM()
userList = parser.getUserList()
itemList = parser.getItemList()

print("First 10 unique users in URM: ", parser.getUserList_unique()[:10])
print("First 10 unique items in URM: ", parser.getItemList_unique()[:10])
