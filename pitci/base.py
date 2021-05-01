import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from typing import Union, Any

from pitci._version import __version__
from pitci.checks import (
    check_type,
    check_attribute,
)
import pitci.nonconformity as nonconformity


class AbsoluteErrorConformalPredictor(ABC):
    """Abstract base class for a conformal interval predictor for any
    underlying  model using non-scaled absolute error as the
    nonconformity measure.
    """

    @abstractmethod
    def __init__(self) -> None:

        self.__version__ = __version__

    def calibrate(
        self,
        data: Any,
        response: Union[np.ndarray, pd.Series],
        alpha: Union[int, float] = 0.95,
    ) -> None:
        """Method to calibrate conformal intervals that will be applied
        to new instances when calling predict_with_interval.

        Method calls _calibrate_interval to set the default (fixed width)
        interval.

        Parameters
        ----------
        data : any
            Dataset to calibrate baselines on.

        alpha : int or float, default = 0.95
            Confidence level for the interval.

        response : np.ndarray, pd.Series or None, default = None
            The associated response values for every record in data.

        """

        check_type(alpha, [int, float], "alpha")

        if not (alpha >= 0 and alpha <= 1):

            raise ValueError("alpha must be in range [0 ,1]")

        if response is None:

            raise TypeError("response cannot be none")

        self._calibrate_interval(data=data, alpha=alpha, response=response)

    def predict_with_interval(self, data: Any) -> np.ndarray:
        """Method to generate predictions on data with conformal intervals.

        This method calls the _generate_predictions method once to
        generate predictions and then puts the half interval calculated
        in _calibrate_interval about the predictions.

        Parameters
        ----------
        data : Any
            Dataset to generate predictions with conformal intervals.

        Returns
        -------
        predictions_with_interval : np.ndarray
            Array of predictions with intervals for each row in data.
            Output array will have 3 columns where the first is the
            lower interval, second are the predictions and the third
            is the upper interval.

        """

        check_attribute(
            self,
            "baseline_interval",
            "AbsoluteErrorConformalPredictor does not have baseline_interval attribute, "
            "run calibrate first.",
        )

        predictions = self._generate_predictions(data)

        n_preds = predictions.shape[0]

        lower_interval = predictions - self.baseline_interval
        upper_interval = predictions + self.baseline_interval

        predictions_with_interval = np.concatenate(
            (
                lower_interval.reshape((n_preds, 1)),
                predictions.reshape((n_preds, 1)),
                upper_interval.reshape((n_preds, 1)),
            ),
            axis=1,
        )

        return predictions_with_interval

    @abstractmethod
    def _generate_predictions(self, data: Any) -> np.ndarray:
        """Method to generate predictions with underlying model.

        Parameters
        ----------
        data : Any
            Dataset to generate predictions for.

        """

        pass

    def _calibrate_interval(
        self,
        data: Any,
        response: Union[np.ndarray, pd.Series],
        alpha: Union[int, float] = 0.95,
    ) -> None:
        """Method to set the baseline conformal interval. Result is stored
        in the baseline_interval attribute.

        The value passed in alpha is also stored in an attribute of the
        same name.

        Parameters
        ----------
        data : Any
            Dataset to use to set baseline interval width.

        alpha : int or float, default = 0.95
            Confidence level for the interval.

        response : np.ndarray, pd.Series or None, default = None
            The response values for the records in data. If passed as
            None then the function will attempt to extract the response from
            the data argument with get_label.

        """

        self.alpha = alpha

        predictions = self._generate_predictions(data)

        nonconformity_values = nonconformity.absolute_error(
            predictions=predictions, response=response
        )

        self.baseline_interval = np.quantile(nonconformity_values, alpha)
