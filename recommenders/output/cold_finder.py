from collections import Counter
import pandas as pd
import numpy as np
from DataUtils.dataLoader import *
import re

urm_all = load_URM_all()
output = "08_01_hyperTuned3"


def load_sample(file_name):
    cols = ['user_id', 'item_list']
    sample_data = pd.read_csv("C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/output/" + file_name + ".csv",  # interactions.csv
                              names=cols, header=0)
    return sample_data


def find_cold_in_output(file_name):
    s = load_sample(file_name)
    x = s.item_list.values

    it = []
    for i in x:
        it.append(re.findall(r'\d+', i))

    flattened = []
    for sublist in it:
        for val in sublist:
            flattened.append(int(val))

    item_pop = np.ediff1d(urm_all.tocsc().indptr)
    cold_items = list(np.where(item_pop == 0)[0])
    print(len(cold_items))

    z = Counter(flattened)
    tot = 0
    for item in cold_items:
        a = z[item]
        tot = tot + a

    # In theory these cold items are items that are found to be recommended to users but there are no interactions of such items in the
    # original URM data !

    print("[ColdFinder] In file " + file_name + " we found a total of = {} cold items".format(tot))


# find_cold_in_output(output)