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

    selected_score = np.quantile(nonconformity_scores, alpha, interpolation="higher")

    return selected_score
