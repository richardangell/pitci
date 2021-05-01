import numpy as np


def absolute_error(predictions, response):
    """Function to calculate absolute error."""

    return np.abs(predictions - response)


def scaled_absolute_error(predictions, response, scaling):
    """Function to apply scaling factor to absolute error."""

    return absolute_error(predictions, response) / scaling
