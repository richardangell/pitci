"""Module containing functions to calculate nonconformity measures."""

import numpy as np


def absolute_error(predictions, response):
    """Function to calculate absolute error."""

    return np.abs(predictions - response)


def scaled_absolute_error(predictions, response, scaling):
    """Function to apply scaling factor to absolute error."""

    return absolute_error(predictions, response) / scaling


def nonconformity_at_alpha(nonconformity_scores, alpha):
    """Function to return the nonconformity score at a given confidence level
    alpha.
    """

    n = nonconformity_scores.shape[0]

    # note, -1 is for 0 based indexing
    alpha_index = int(np.floor(alpha * (n + 1))) - 1

    alpha_index = np.clip(alpha_index, a_min=0, a_max=n - 1)

    selected_score = np.sort(nonconformity_scores)[alpha_index]

    return selected_score
