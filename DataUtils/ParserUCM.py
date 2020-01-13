from sklearn import preprocessing
import numpy as np
import scipy.sparse as sps

def ucm_all_builder(urm_all, ucm_age_tuples, ucm_region_tuples, ucm_interactions_tuples):
    print("[UCM ALL BUILDER] Starting")
    row_age, column_age, data_age = zip(*ucm_age_tuples)
    row_region, column_region, data_region = zip(*ucm_region_tuples)
    row_interactions, column_interactions, data_interactions = zip(*ucm_interactions_tuples)

    le_age = preprocessing.LabelEncoder()
    le_age.fit(data_age)
    data_age = le_age.transform(data_age)

    le_region = preprocessing.LabelEncoder()
    le_region.fit(data_region)
    data_region = le_region.transform(data_region)

    n_users = urm_all.shape[0]
    n_features_ucm_age = max(data_age) + 1
    n_features_ucm_region = max(data_region) + 1
    n_features_ucm_interactions = max(column_interactions) + 1

    ucm_age_shape = (n_users, n_features_ucm_age)
    ucm_region_shape = (n_users, n_features_ucm_region)
    ucm_interactions_shape = (n_users, n_features_ucm_interactions)

    ones_ucm_age = np.ones(len(data_age))
    ones_ucm_region = np.ones(len(data_region))

    ucm_age = sps.coo_matrix((ones_ucm_age, (row_age, data_age)), shape=ucm_age_shape)
    ucm_region = sps.coo_matrix((ones_ucm_region, (row_region, data_region)), shape=ucm_region_shape)
    ucm_interactions = sps.coo_matrix((data_interactions, (row_interactions, column_interactions)), shape=ucm_interactions_shape)

    ucm_all = sps.hstack((ucm_age, ucm_region))
    ucm_all = sps.hstack((ucm_all, ucm_interactions))
    ucm_all = ucm_all.tocsr()

    return ucm_all
