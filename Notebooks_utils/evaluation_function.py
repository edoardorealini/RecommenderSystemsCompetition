#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21/10/2018

@author: Maurizio Ferrari Dacrema
"""
from time import sleep

import numpy as np
import progressbar
import scipy.sparse as sps
from tqdm import tqdm

from DataUtils.datasetSplitter import *


def precision(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score



def recall(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score



def MAP(is_relevant, relevant_items):

    #is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score

def evaluate_algorithm_original(URM_test, recommender_object, at=10):
    print("Evaluating algorithm with original evaluator")
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    URM_test = sps.csr_matrix(URM_test)

    n_users = URM_test.shape[0]


    for user_id in tqdm(range(n_users)):

        start_pos = URM_test.indptr[user_id]
        end_pos = URM_test.indptr[user_id+1]

        if end_pos-start_pos>0:

            relevant_items = URM_test.indices[start_pos:end_pos]

            recommended_items = recommender_object.recommend(user_id, at=at)
            num_eval+=1

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            cumulative_precision += precision(is_relevant, relevant_items)
            cumulative_recall += recall(is_relevant, relevant_items)
            cumulative_MAP += MAP(is_relevant, relevant_items)


    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP))

    result_dict = {
        "precision": cumulative_precision,
        "recall": cumulative_recall,
        "MAP": cumulative_MAP,
    }

    return result_dict


def evaluate_algorithm(URM_test, recommender_object, userList = getUserList(), at=10):

    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    URM_test = sps.csr_matrix(URM_test)

    n_users = URM_test.shape[0]

    print("[evaluation] length of user list: ", len(userList))

    for user_id in tqdm(userList):

        #if user_id % 10000 == 0:
        #    print("Evaluated user {} of {}".format(user_id, n_users))

        if user_id != 30910:
            start_pos = URM_test.indptr[user_id]
            end_pos = URM_test.indptr[user_id + 1]
        else:
            start_pos = URM_test.indptr[user_id]

        if end_pos-start_pos > 0:
            if user_id == 30910:
                relevant_items = URM_test.indices[start_pos:]
            else:
                relevant_items = URM_test.indices[start_pos:end_pos]

            recommended_items = recommender_object.recommend(user_id)
            num_eval += 1

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            cumulative_precision += precision(is_relevant, relevant_items)
            cumulative_recall += recall(is_relevant, relevant_items)
            cumulative_MAP += MAP(is_relevant, relevant_items)


    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP))

    result_dict = {
        "precision": cumulative_precision,
        "recall": cumulative_recall,
        "MAP": cumulative_MAP,
    }

    return result_dict

# update:  automatically getting the user list
def evaluate_algorithm_coldUsers(URM_test, recommender_object, recommender_cold, userList = getUserList(), at=5):

    coldUserList = getColdUsers()

    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    URM_test = sps.csr_matrix(URM_test)

    n_users = URM_test.shape[0]

    print("[evaluation] length of user list: ", len(userList))

    for user_id in tqdm(userList):

        #if user_id % 10000 == 0:
        #    print("Evaluated user {} of {}".format(user_id, n_users))

        if user_id != 30910:
            start_pos = URM_test.indptr[user_id]
            end_pos = URM_test.indptr[user_id + 1]
        else:
            start_pos = URM_test.indptr[user_id]

        if end_pos-start_pos > 0:
            if user_id == 30910:
                relevant_items = URM_test.indices[start_pos:]
            else:
                relevant_items = URM_test.indices[start_pos:end_pos]

            if user_id in coldUserList:
                recommended_items = recommender_cold.recommend(user_id)

            else:
                recommended_items = recommender_object.recommend(user_id)

            num_eval += 1

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            cumulative_precision += precision(is_relevant, relevant_items)
            cumulative_recall += recall(is_relevant, relevant_items)
            cumulative_MAP += MAP(is_relevant, relevant_items)


    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP))

    result_dict = {
        "precision": cumulative_precision,
        "recall": cumulative_recall,
        "MAP": cumulative_MAP,
    }

    return result_dict