import random
import pandas as pd
import numpy as np
from tqdm import tqdm

import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler

def implicit_als(sparse_data, alpha_val=40, iterations=10, lambda_val=0.1, features=10):


    """ Implementation of Alternating Least Squares with implicit data. We iteratively
        compute the user (x_u) and item (y_i) vectors using the following formulas:

        x_u = ((Y.T*Y + Y.T*(Cu - I) * Y) + lambda*I)^-1 * (X.T * Cu * p(u))
        y_i = ((X.T*X + X.T*(Ci - I) * X) + lambda*I)^-1 * (Y.T * Ci * p(i))

        Args:
            sparse_data (csr_matrix): Our sparse user-by-item matrix

            alpha_val (int): The rate in which we'll increase our confidence
            in a preference with more interactions.

            iterations (int): How many times we alternate between fixing and
            updating our user and item vectors

            lambda_val (float): Regularization value

            features (int): How many latent features we want to compute.

        Returns:
            X (csr_matrix): user vectors of size users-by-features

            Y (csr_matrix): item vectors of size items-by-features
         """

    # Calculate the foncidence for each value in our data
    confidence = sparse_data * alpha_val

    # Get the size of user rows and item columns
    user_size, item_size = sparse_data.shape

    # We create the user vectors X of size users-by-features, the item vectors
    # Y of size items-by-features and randomly assign the values.
    X = sparse.csr_matrix(np.random.normal(size=(user_size, features)))
    Y = sparse.csr_matrix(np.random.normal(size=(item_size, features)))

    # Precompute I and lambda * I
    X_I = sparse.eye(user_size)
    Y_I = sparse.eye(item_size)

    I = sparse.eye(features)
    lI = lambda_val * I

    """ Continuation of implicit_als function"""

    # Start main loop. For each iteration we first compute X and then Y
    for i in range(iterations):
        print
        'iteration %d of %d' % (i + 1, iterations)

        # Precompute Y-transpose-Y and X-transpose-X
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)

        # Loop through all users
        for u in tqdm(range(user_size)):
            # Get the user row.
            u_row = confidence[u, :].toarray()

            # Calculate the binary preference p(u)
            p_u = u_row.copy()
            p_u[p_u != 0] = 1.0

            # Calculate Cu and Cu - I
            CuI = sparse.diags(u_row, [0])
            Cu = CuI + Y_I

            # Put it all together and compute the final formula
            yT_CuI_y = Y.T.dot(CuI).dot(Y)
            yT_Cu_pu = Y.T.dot(Cu).dot(p_u.T)
            X[u] = spsolve(yTy + yT_CuI_y + lI, yT_Cu_pu)

        for i in range(item_size):
            # Get the item column and transpose it.
            i_row = confidence[:, i].T.toarray()

            # Calculate the binary preference p(i)
            p_i = i_row.copy()
            p_i[p_i != 0] = 1.0

            # Calculate Ci and Ci - I
            CiI = sparse.diags(i_row, [0])
            Ci = CiI + X_I

            # Put it all together and compute the final formula
            xT_CiI_x = X.T.dot(CiI).dot(X)
            xT_Ci_pi = X.T.dot(Ci).dot(p_i.T)
            Y[i] = spsolve(xTx + xT_CiI_x + lI, xT_Ci_pi)

    return X, Y

from DataUtils.dataLoader import *

train, test = load_data_split(0)

user_vecs, item_vecs = implicit_als(train, iterations=20, features=20, alpha_val=40)