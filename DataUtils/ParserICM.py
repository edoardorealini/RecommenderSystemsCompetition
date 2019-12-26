from sklearn import preprocessing
import numpy as np
import scipy.sparse as sps

def icm_all_builder(urm_all, icm_asset_tuples, icm_price_tuples, icm_sub_class_tuples):
    print("[ICM ALL BUILDER] Starting")
    row_asset, column_asset, data_asset = zip(*icm_asset_tuples)
    row_price, column_price, data_price = zip(*icm_price_tuples)
    row_sub_class, column_sub_class, data_sub_class = zip(*icm_sub_class_tuples)

    le_asset = preprocessing.LabelEncoder()
    le_asset.fit(data_asset)
    data_asset = le_asset.transform(data_asset)

    le_price = preprocessing.LabelEncoder()
    le_price.fit(data_price)
    data_price = le_price.transform(data_price)

    n_items = urm_all.shape[1]
    n_features_icm_asset = max(data_asset) + 1
    n_features_icm_price = max(data_price) + 1
    n_features_icm_sub_class = max(column_sub_class) + 1

    icm_asset_shape = (n_items, n_features_icm_asset)
    icm_price_shape = (n_items, n_features_icm_price)
    icm_sub_class_shape = (n_items, n_features_icm_sub_class)

    ones_icm_asset = np.ones(len(data_asset))
    ones_icm_price = np.ones(len(data_price))

    icm_asset = sps.coo_matrix((ones_icm_asset, (row_asset, data_asset)), shape=icm_asset_shape)
    icm_price = sps.coo_matrix((ones_icm_price, (row_price, data_price)), shape=icm_price_shape)
    icm_sub_class = sps.coo_matrix((data_sub_class, (row_sub_class, column_sub_class)), shape=icm_sub_class_shape)

    icm_all = sps.hstack((icm_asset, icm_price))
    icm_all = sps.hstack((icm_all, icm_sub_class))
    icm_all = icm_all.tocsr()

    return icm_all
