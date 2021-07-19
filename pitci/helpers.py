"""Module containing functions used for evaluating interval regions."""

import pandas as pd
import numpy as np

from typing import Union, List, Tuple, Optional

from .checks import check_type


def gather_intervals(
    lower_interval: Optional[Union[np.ndarray, pd.Series]] = None,
    upper_interval: Optional[Union[np.ndarray, pd.Series]] = None,
    intervals_with_predictions: Optional[np.ndarray] = None,
) -> Tuple[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.Series]]:
    """Function to perform checks on passed intervals and return lower and upper
    intervals separately if they are passed combined in intervals_with_predictions.
    """

    if (
        (lower_interval is None and intervals_with_predictions is None)
        or (upper_interval is None and intervals_with_predictions is None)
        or (
            upper_interval is None
            and lower_interval is None
            and intervals_with_predictions is None
        )
    ):

        raise ValueError(
            "either lower_interval and upper_interval or intervals_with_predictions must"
            "be specified but both are None"
        )

    if (
        (lower_interval is not None and intervals_with_predictions is not None)
        or (upper_interval is not None and intervals_with_predictions is not None)
        or (
            upper_interval is not None
            and lower_interval is not None
            and intervals_with_predictions is not None
        )
    ):

        raise ValueError(
            "either lower_interval and upper_interval or intervals_with_predictions must"
            "be specified but both are specified"
        )

    # if intervals_with_predictions is passed, split out the first and third columns
    # into lower_interval and upper_interval
    if intervals_with_predictions is not None:

        check_type(
            intervals_with_predictions, [np.ndarray], "intervals_with_predictions"
        )

        if not intervals_with_predictions.shape[1] == 3:
            raise ValueError("expecting intervals_with_predictions to have 3 columns")

        lower_interval_return = intervals_with_predictions[:, 0]
        upper_interval_return = intervals_with_predictions[:, 2]

    else:

        lower_interval_return = lower_interval
        upper_interval_return = upper_interval

    check_type(lower_interval_return, [np.ndarray, pd.Series], "lower_interval_return")
    check_type(upper_interval_return, [np.ndarray, pd.Series], "upper_interval_return")

    if lower_interval_return.shape[0] != upper_interval_return.shape[0]:

        raise ValueError(
            "lower_interval_return and upper_interval_return have different shapes"
        )

    return lower_interval_return, upper_interval_return


def check_response_within_interval(
    response: Union[np.ndarray, pd.Series],
    lower_interval: Optional[Union[np.ndarray, pd.Series]] = None,
    upper_interval: Optional[Union[np.ndarray, pd.Series]] = None,
    intervals_with_predictions: Optional[np.ndarray] = None,
) -> pd.Series:
    """Function to check the number of times a response lies within
    a prediction interval.

    Either both lower_interval and upper_interval or intervals_with_predictions
    must be specified.

    The function returns the proportion of the response that lies between
    the intervals.

    Parameters
    ----------
    response : np.ndarray, pd.Series
        Response or actual values corresponding to each row in the passed
        intervals.

    lower_interval : np.ndarray, pd.Series or None, default = None
        Lower intervals, if None then lower interval will be taken from the
        first column in intervals_with_predictions.

    upper_interval : np.ndarray, pd.Series or None, default = None
        Upper intervals, if None then upper interval will be taken from the
        first column in intervals_with_predictions.

    intervals_with_predictions : np.ndarry or None, default = None
        Lower intervals and upper intervals combined in a single np array.
        The array must have 3 columns. The lower interval is assumed to be
        the first column and the upper column is assumed to be the third
        column.

    """

    lower_interval, upper_interval = gather_intervals(
        lower_interval=lower_interval,
        upper_interval=upper_interval,
        intervals_with_predictions=intervals_with_predictions,
    )

    check_type(response, [np.ndarray, pd.Series], "response")

    if not response.shape[0] == lower_interval.shape[0]:
        raise ValueError("response and intervals have different numbers of rows")

    response_within_interval = (response >= lower_interval) & (
        response <= upper_interval
    )

    results = pd.Series(response_within_interval).value_counts() / response.shape[0]

    return results


def check_interval_width(
    lower_interval: Optional[Union[np.ndarray, pd.Series]] = None,
    upper_interval: Optional[Union[np.ndarray, pd.Series]] = None,
    intervals_with_predictions: Optional[np.ndarray] = None,
    quantiles: List[Union[int, float]] = [
        0,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        1,
    ],
) -> pd.Series:
    """Function to check the distribution of prediction intervals.

    Either both lower_interval and upper_interval or intervals_with_predictions
    must be specified.

    The specified quantiles of the interval distribution, the mean, std and iqr
    are returned in a dict. A histogram of the distribution is also printed.

    Parameters
    ----------
    lower_interval : np.ndarray, pd.Series or None, default = None
        Lower intervals, if None then lower interval will be taken from the
        first column in intervals_with_predictions.

    upper_interval : np.ndarray, pd.Series or None, default = None
        Upper intervals, if None then upper interval will be taken from the
        first column in intervals_with_predictions.

    intervals_with_predictions : np.ndarry or None, default = None
        Lower intervals and upper intervals combined in a single np array.
        The array must have 3 columns. The lower interval is assumed to be
        the first column and the upper column is assumed to be the third
        column.

    quantiles : list
        List of quantiles to report on the distribution of the interval widths.

    """

    lower_interval, upper_interval = gather_intervals(
        lower_interval=lower_interval,
        upper_interval=upper_interval,
        intervals_with_predictions=intervals_with_predictions,
    )

    interval_width = pd.Series(upper_interval - lower_interval)

    interval_width_distribution = interval_width.quantile(quantiles)

    interval_width_distribution["mean"] = interval_width.mean()
    interval_width_distribution["std"] = interval_width.std()
    interval_width_distribution["iqr"] = interval_width.quantile(
        0.75
    ) - interval_width.quantile(0.25)

    return interval_width_distribution


def prepare_prediction_interval_df(
    intervals_with_predictions: np.ndarray, response: pd.Series
) -> pd.DataFrame:
    """Put response column and n x 3 array into a pd.DataFrame with columns;
    "lower", "predictions", "upper" and response".

    Parameters
    ----------
    intervals_with_predictions : np.ndarray
        n by 3 array containing lower interval values, predictions and upper
        interval values. The columns will be added to output in columns;
        "lower", "predictions" and "upper".

    response : pd.Series or np.ndarray
        Response column to be added to output, in "response" column. Must have
        the same number of rows as intervals_with_predictions.

    Returns
    -------
    df : pd.DataFrame
        4 column pd.DataFrame containing values passed in intervals_with_predictions
        and response with columns; "lower", "predictions", "upper" and response".

    """

    check_type(intervals_with_predictions, [np.ndarray], "intervals_with_predictions")

    check_type(response, [np.ndarray, pd.Series], "response")

    if intervals_with_predictions.shape[1] != 3:

        raise ValueError("intervals_with_predictions must have 3 columns")

    if intervals_with_predictions.shape[0] != response.shape[0]:

        raise ValueError(
            "intervals_with_predictions and response have different numbers of rows"
        )

    df = pd.DataFrame(
        intervals_with_predictions, columns=["lower", "prediction", "upper"]
    )

    if type(response) is pd.Series:

        df["response"] = response.values

    else:

        df["response"] = response

    return df


def create_interval_buckets(
    intervals_with_predictions: pd.DataFrame, cut_function: str = "qcut", **kwargs
) -> pd.DataFrame:
    """Function to create a new column in a DataFrame that buckets all rows
    on the widthof the intervals in the DataFrame.

    Parameters
    ----------
    intervals_with_predictions : pd.DataFrame
        Data to add column too containing buckets of interval widths. Must
        have columns called "upper" and "lower" that gives the limits
        of the intervals for each row.

    cut_function : str
        Type of bucketing to use, must be either cut or qcut. Decides
        the pandas cut function to use.

    **kwargs : any
        Arbitrary keyword arguments to pass onto the pandas cut method.

    Returns
    -------
    intervals_with_predictions : pd.DataFrame
        Input data with new column called "interval_width_bucket" that
        splits the data on the width of the intervals in the data (defined
        by the "lower" and "upper" columns)

    """

    check_type(intervals_with_predictions, [pd.DataFrame], "intervals_with_predictions")

    check_type(cut_function, [str], "cut_function")

    if cut_function not in ["qcut", "cut"]:

        raise ValueError("cut_function must be either qcut or cut")

    interval_width = (
        intervals_with_predictions["upper"] - intervals_with_predictions["lower"]
    )

    if cut_function == "qcut":

        intervals_with_predictions["interval_width_bucket"] = pd.qcut(
            x=interval_width, **kwargs
        )

    else:

        intervals_with_predictions["interval_width_bucket"] = pd.cut(
            x=interval_width, **kwargs
        )

    return intervals_with_predictions
