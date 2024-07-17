"""
Code modified from the original Matlab's implementation of CPM.
https://github.com/YaleMRRC/CPM
"""

import numpy as np
import scipy.io
from scipy.stats import pearsonr


def corr(X, y):
    """
    Inupts:
        X - [feature_dim x n_subjects]
        y - [n_subjects x 1]
    Outputs:
        r_mat - [feature_dim]
        p_mat - [feature_dim]
    """
    r_mat = np.zeros(X.shape[0])
    p_mat = np.zeros(X.shape[0])
    for i, x in enumerate(X):
        r_val, p_val = scipy.stats.pearsonr(x, y)
        r_mat[i] = r_val
        p_mat[i] = p_val
    return r_mat, p_mat


def cpm(edge_weights, labels, traintestid):
    """
    This function evaluates the edge weigths and returns (also prints) the R value by CPM
    """
    train_id = traintestid["train_id"]
    test_id = traintestid["test_id"]
    all_mats = edge_weights
    all_behav = labels

    final_test_mats = all_mats[:, test_id]
    final_test_behave = all_behav[test_id]

    all_mats = all_mats[:, train_id]
    all_behav = all_behav[train_id]

    # threshold for feature selection
    thresh = 0.05

    # correlate all edges with behavior
    r_mat, p_mat = corr(all_mats, all_behav)

    final_mask = np.zeros(all_mats.shape[0])
    final_edges = np.where((r_mat > 0) & (p_mat < thresh))[0]
    final_mask[final_edges] = 1

    # first get the train linear model
    train_sum = np.sum(all_mats * final_mask[:, np.newaxis], axis=0)

    fit = np.polyfit(train_sum, all_behav, 1)

    # evaluate on test
    test_sum = np.sum(final_test_mats * final_mask[:, np.newaxis], axis=0)
    test_behav = fit[0] * test_sum + fit[1]

    test_R, test_P = pearsonr(test_behav, final_test_behave)

    print("R: ", test_R)
    return test_R, test_P
