from sklearn import preprocessing
import numpy as np
import scipy.sparse as sps

def ucm_all_builder(urm_all, ucm_age_tuples, ucm_region_tuples, ucm_interactions_tuples):
    print("[UCM ALL BUILDER] Starting")
    row_age, column_age, data_age = zip(*ucm_age_tuples)
    row_region, column_region, data_region = zip(*ucm_region_tuples)
    row_interactions, column_interactions, data_interactions = zip(*ucm_interactions_tuples)

    le_age = preprocessing.LabelEncoder()
    le_age.fit(column_age)
    column_age = le_age.transform(column_age)

    le_region = preprocessing.LabelEncoder()
    le_region.fit(column_region)
    column_region = le_region.transform(column_region)

    le_interactions = preprocessing.LabelEncoder()
    le_interactions.fit(column_interactions)
    column_interactions = le_interactions.transform(column_interactions)

    n_users = urm_all.shape[0]
    n_features_ucm_age = max(column_age) + 1
    n_features_ucm_region = max(column_region) + 1
    n_features_ucm_interactions = max(column_interactions) + 1

    ucm_age_shape = (n_users, n_features_ucm_age)
    ucm_region_shape = (n_users, n_features_ucm_region)
    ucm_interactions_shape = (n_users, n_features_ucm_interactions)

    ucm_age = sps.coo_matrix((data_age, (row_age, column_age)), shape=ucm_age_shape)
    ucm_region = sps.coo_matrix((data_region, (row_region, column_region)), shape=ucm_region_shape)
    ucm_interactions = sps.coo_matrix((data_interactions, (row_interactions, column_interactions)), shape=ucm_interactions_shape)

    ucm_all = sps.hstack((ucm_age, ucm_region))
    ucm_all = sps.hstack((ucm_all, ucm_interactions))
    ucm_all = ucm_all.tocsr()

    return ucm_all
