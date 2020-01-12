from lightfm import LightFM
from lightfm.evaluation import auc_score
import numpy as np

from DataUtils.dataLoader import *

NUM_THREADS = 1
NUM_COMPONENTS = 30
NUM_EPOCHS = 3
ITEM_ALPHA = 1e-6

train, test = load_data_split(0)
item_features = load_ICM()

model = LightFM(loss='warp',
                item_alpha=ITEM_ALPHA,
                no_components=NUM_COMPONENTS)

# Fit the hybrid model. Note that this time, we pass
# in the item features matrix.
model = model.fit(train,
                item_features=item_features,
                epochs=NUM_EPOCHS,
                num_threads=NUM_THREADS)

# Don't forget the pass in the item features again!
train_auc = auc_score(model,
                      train,
                      item_features=item_features,
                      num_threads=NUM_THREADS).mean()
print('Hybrid training set AUC: %s' % train_auc)


test_auc = auc_score(model,
                    test,
                    train_interactions=train,
                    item_features=item_features,
                    num_threads=NUM_THREADS).mean()
print('Hybrid test set AUC: %s' % test_auc)


def sample_recommendation(model, train, item_features, user_ids):
    n_users, n_items = train.shape

    for user_id in user_ids:
        known_positives = item_features[train.tocsr()[user_id].indices]

        scores = model.predict(user_id, np.arange(n_items))
        top_items = item_features[np.argsort(-scores)]

        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)

sample_recommendation(model, train, item_features.T, [3, 25, 450])