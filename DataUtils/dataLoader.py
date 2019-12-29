import scipy.sparse as sps


def load_all_data():
    URM_all = sps.load_npz(
        'C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/sparse_URM.npz')
    print("URM correctly loaded from file: data/competition/sparse_URM.npz")
    URM_all = URM_all.tocsr()

    URM_test = sps.load_npz(
        'C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/URM_test.npz')
    print("URM_test correctly loaded from file: data/competition/URM_test.npz")
    URM_test = URM_test.tocsr()

    URM_train = sps.load_npz(
        'C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/URM_train.npz')
    print("URM_train correctly loaded from file: data/competition/URM_train.npz")
    URM_train = URM_train.tocsr()

    return URM_all, URM_train, URM_test


def load_ICM():
    ICM_all = sps.load_npz('C:/Users/Utente/Desktop/RecSys-Competition-2019/recommenders/data/competition/sparse_ICM.npz')
    print("ICM_all correctly loaded from path: data/competition/sparse_ICM.npz")
    ICM_all = ICM_all.tocsr()

    return ICM_all

