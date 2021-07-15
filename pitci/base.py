"""
Module containing base conformal predictor classes, model-type specific conformal
predictor classes will inherit from these base classes.
"""

import pandas as pd
import numpy as np
import warnings
from abc import ABC, abstractmethod

from typing import Union, Any, List, Dict

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

    Parameters
    ----------
    model : Any
        Underly model to generate prediction intervals for.

    """

    @abstractmethod
    def __init__(self, model: Any) -> None:

        self.__version__ = __version__
        self.model = model

    def calibrate(
        self,
        data: Any,
        response: Union[np.ndarray, pd.Series],
        alpha: Union[int, float] = 0.95,
    ) -> None:
        """Method to calibrate conformal intervals that will be applied
        to new instances when calling `predict_with_interval`.

        Method calls `_calibrate_interval` to set the default, fixed width,
        interval.

        Parameters
        ----------
        data : Any
            Dataset to calibrate baselines on.

        response : np.ndarray or pd.Series
            The associated response values for every record in `data`.

        alpha : int or float, default = 0.95
            Confidence level for the interval.

        """

        check_type(alpha, [int, float], "alpha")
        check_type(response, [np.ndarray, pd.Series], "response")

        if not (alpha >= 0 and alpha <= 1):

            raise ValueError("alpha must be in range [0 ,1]")

        self._calibrate_interval(data=data, alpha=alpha, response=response)

    def predict_with_interval(self, data: Any) -> np.ndarray:
        """Method to generate predictions on data with conformal intervals.

        This method calls the `_generate_predictions` method once to
        generate predictions and then puts the half interval calculated
        in `_calibrate_interval` about the predictions.

        Parameters
        ----------
        data : Any
            Dataset on which to generate predictions with conformal intervals.

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
        in the `baseline_interval` attribute.

        The value passed in `alpha` is also stored in an attribute of the
        same name.

        Parameters
        ----------
        data : Any
            Dataset to use to set baseline interval width.

        response : np.ndarray or pd.Series
            The response values for the records in `data`.

        alpha : int or float, default = 0.95
            Confidence level for the interval.

        """

        self.alpha = alpha

        predictions = self._generate_predictions(data)

        nonconformity_values = nonconformity.absolute_error(
            predictions=predictions, response=response
        )

        self.baseline_interval = nonconformity.nonconformity_at_alpha(
            nonconformity_values, alpha
        )


class LeafNodeScaledConformalPredictor(ABC):
    """Conformal interval predictor for an underlying xgboost model
    using scaled absolute error as the nonconformity measure.

    Class implements inductive conformal intervals where a calibration
    dataset is used to learn the information that is used when generating
    intervals for new instances.

    The predictor outputs varying width intervals for every new instance.
    The scaling function uses the number of times that the leaf nodes were
    visited for each tree in making the prediction, for that row, were
    visited in the calibration dataset.

    Intuitively, for rows that have higher leaf node counts from the calibration
    set - the model will be more 'familiar' with hence the interval for
    these rows will be shrunk. The inverse is true for rows that have lower
    leaf node counts from the calibration set.

    Attributes
    ----------
    leaf_node_counts : list
        Counts of number of times each leaf node in each tree was visited when
        making predictions on the calibration dataset. Attribute is set by the
        _calibrate_leaf_node_counts, called by the calibrate method. The
        length of the list corresponds to the number of trees.

    baseline_interval : float
        Default, baseline conformal interval width. Will be scaled for each
        prediction generated. Attribute is set when _calibrate_interval is
        called by the calibrate method.

    alpha : int or float
        The confidence level of the conformal intervals that will be produced.
        Attribute is set when _calibrate_interval is called by the calibrate
        method.

    """

    leaf_node_counts: list

    @abstractmethod
    def __init__(self, model: Any) -> None:

        self.__version__ = __version__
        self.model = model

    def calibrate(
        self,
        data: Any,
        response: Union[np.ndarray, pd.Series],
        alpha: Union[int, float] = 0.95,
    ) -> None:
        """Method to calibrate conformal intervals that will allow
        prediction intervals that vary by row.

        Method calls _calibrate_leaf_node_counts to record the number
        of times each leaf node is visited across the whole of the
        passed data.

        Method calls _calibrate_interval to set the default interval that
        will be scaled using the inverse of the noncomformity function
        when making predictions. This allows intervals to vary by instance.

        Parameters
        ----------
        data : xgb.DMatrix
            Dataset to use to set baselines.

        alpha : int or float, default = 0.95
            Confidence level for the interval.

        response : np.ndarray or pd.Series
            The response values for the records in data.

        """

        check_type(response, [pd.Series, np.ndarray], "response")
        check_type(alpha, [int, float], "alpha")

        if not (alpha >= 0 and alpha <= 1):

            raise ValueError("alpha must be in range [0 ,1]")

        self._calibrate_leaf_node_counts(data=data)
        self._calibrate_interval(data=data, alpha=alpha, response=response)

    def predict_with_interval(self, data: Any) -> np.ndarray:
        """Method to generate predictions on data with conformal intervals.

        Each prediction is produced with an associated conformal interval.
        The default interval is of a fixed width and this is scaled
        differently for each row. Scaling is done, for a given row, by
        counting the number of times each leaf node, visited to make the
        prediction, was visited in the calibration dataset. The counts of
        leaf node visits in the calibration data are set by the
        _calibrate_leaf_node_counts method.

        Method multiplies the scaling factors, generated by _calculate_scaling_factors
        method, by the baseline_interval value.

        Parameters
        ----------
        data : Any
            Data to generate predictions with conformal intervals on.

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
            "LeafNodeScaledConformalPredictor does not have baseline_interval attribute, "
            "run calibrate first.",
        )

        predictions = self._generate_predictions(data)

        n_preds = predictions.shape[0]

        scaling_factors = self._calculate_scaling_factors(data)

        lower_interval = predictions - (self.baseline_interval * scaling_factors)
        upper_interval = predictions + (self.baseline_interval * scaling_factors)

        predictions_with_interval = np.concatenate(
            (
                lower_interval.reshape((n_preds, 1)),
                predictions.reshape((n_preds, 1)),
                upper_interval.reshape((n_preds, 1)),
            ),
            axis=1,
        )

        return predictions_with_interval

    def _calibrate_interval(
        self,
        data: Any,
        response: Union[np.ndarray, pd.Series],
        alpha: Union[int, float] = 0.95,
    ) -> None:
        """Method to set the baseline conformal interval.

        This is the default interval that will be scaled for differently
        for each row.

        Result is stored in the baseline_interval attribute.

        The value passed in alpha is also stored in an attribute of the
        same name.

        Parameters
        ----------
        data : Any
            Dataset to use to set baseline interval width.

        alpha : int or float, default = 0.95
            Confidence level for the interval.

        response : np.ndarray or pd.Series
            The response values for the records in data.

        """

        self.alpha = alpha

        predictions = predictions = self._generate_predictions(data)

        scaling_factors = self._calculate_scaling_factors(data)

        nonconformity_values = nonconformity.scaled_absolute_error(
            predictions=predictions, response=response, scaling=scaling_factors
        )

        self.baseline_interval = nonconformity.nonconformity_at_alpha(
            nonconformity_values, alpha
        )

    def _calculate_scaling_factors(self, data: Any) -> np.ndarray:
        """Method to calculate the scaling factors for a given dataset.

        First leaf node indexes are generated for the passed data using
        the _generate_leaf_node_predictions method.

        Then leaf node indexes are passed to _count_leaf_node_visits_from_calibration
        which, for each row, counts the total number of times each leaf
        node index was visited in the calibration dataset.

        1 / leaf node counts are returned from this function so that the scaling
        factor is inverted i.e. smaller values are better.

        Parameters
        ----------
        data : Any
            Data to calculate interval scaling factors for.

        Returns
        -------
        leaf_node_counts : np.ndarray
            Array of same length as input data giving factor for each input row.

        """

        leaf_node_predictions = self._generate_leaf_node_predictions(data)

        leaf_node_counts = self._count_leaf_node_visits_from_calibration(
            leaf_node_predictions=leaf_node_predictions
        )

        # change scaling factor to be; the smaller the better
        reciprocal_leaf_node_counts = 1 / leaf_node_counts

        return reciprocal_leaf_node_counts

    def _count_leaf_node_visits_from_calibration(
        self, leaf_node_predictions: np.ndarray
    ) -> np.ndarray:
        """Method to count the number of times each leaf node
        was visited across each tree in the calibration dataset.

        The function _sum_dict_values is applied to each row in
        leaf_node_predictions, passing the leaf_node_counts attribute
        in the counts arg. leaf_node_counts contains the number of
        times each leaf node in each tree was visited in making predictions
        for the calibration dataset (set in by the _calibrate_leaf_node_counts
        method).

        Parameters
        ----------
        leaf_node_predictions : np.ndarray
            Array output from the relevant underlying model predict method
            which produces the leaf node visited in each tree for each
            row of data scored.

        """

        check_attribute(
            self,
            "leaf_node_counts",
            "leaf_node_counts attribute missing, run calibrate first.",
        )

        leaf_node_counts = np.apply_along_axis(
            _sum_dict_values,
            1,
            leaf_node_predictions,
            counts=self.leaf_node_counts,
        )

        return leaf_node_counts

    def _calibrate_leaf_node_counts(self, data: Any) -> None:
        """Method to set that baseline leaf node counts.

        First the _generate_leaf_node_predictions method is called to
        get the leaf node indexes that were visted in every tree for
        every row in the passed data arg.

        Then each column in the output from _generate_leaf_node_predictions
        (representing a single tree in the model) is the tabulated to
        count the number of times each leaf node in the tree was
        visited when making predictions for data.

        Parameters
        ----------
        data : Any
            Data to set baseline counts of leaf nodes.

        """

        leaf_node_predictions = self._generate_leaf_node_predictions(data)

        leaf_node_predictions_df = pd.DataFrame(leaf_node_predictions)

        self.leaf_node_counts = []

        for tree_no, column in enumerate(leaf_node_predictions_df.columns.values):

            # count the number of times each leaf node is visited in
            # each tree for predictions on data
            self.leaf_node_counts.append(
                leaf_node_predictions_df[tree_no].value_counts().to_dict()
            )

    @abstractmethod
    def _generate_predictions(self, data: Any) -> np.ndarray:
        """Method to generate predictions with underlying model.

        Parameters
        ----------
        data : Any
            Data to generate predictions on.

        """

        pass

    @abstractmethod
    def _generate_leaf_node_predictions(self, data: Any) -> np.ndarray:
        """Method to generate leaf node predictions with underlying model.

        Parameters
        ----------
        data : Any
            Data to generate predictions on.

        """

        pass


def _sum_dict_values(arr: np.ndarray, counts: List[Dict[int, int]]) -> int:
    """Function to sum values in a list of dictionaries
    where the key to sum from each dict is defined by the
    elements of arr.

    Function iterates over each element in the array (which
    is a leaf node index for each tree in the model) and sums
    the value in the counts list for that leaf node index
    in that tree.

    The counts list must have length n when n is the length
    of the arr arg. Each item in the list gives the counts of the
    number of times each leaf node in the given tree was visited
    when making predictions on the calibration dataset.

    Parameters
    ----------
    arr : np.ndarry
        Single row of an array containing leaf node indexes.

    counts : dict
        Counts of the number of times each leaf node in each
        tree was visited when making predictions on the
        calibration dataset.

    """

    total = 0

    for i, value in enumerate(arr):

        tree_counts = counts[i]

        try:

            total += tree_counts[value]

        # if value is not in the keys of tree_counts then we simply
        # move on, this means that that particular leaf node was not
        # visited in the calibration
        # it is not guaranteed that every leaf node will be visited
        # unless the same dataset that was used for training was
        # used for calibration
        except KeyError:

            pass

    return total


class SplitConformalPredictor(LeafNodeScaledConformalPredictor):
    """Abstract base class for conformal interval predictor whose
    intervals vary using a scaling factor based off the number of
    times leaf nodes used to make predictions were visited on a
    calibration set. Intervals are also split into bins by the scaling
    factor and calibrated separately for each bin.

    Class implements inductive conformal intervals where a calibration
    dataset is used to learn the information that is used when generating
    intervals for new instances.

    Attributes
    ----------
    n_bins : int
        Number of bins to split data based off the scaling factors.

    bin_quantiles : float
        Quantiles of the scaling factor values that will be used to define
        the limits of the bins.

    baseline_intervals : list
        Baseline interval for each bin. When intervals are generated the
        appropriate baseline interval will be selected given the scaling
        factor value, this is then scaled by the scaling factor.

    """

    def __init__(self, model: Any, n_bins: int = 3) -> None:

        check_type(n_bins, [int], "n_bins")

        if not n_bins > 1:

            raise ValueError("n_bins should be greater than 1")

        self.n_bins = n_bins
        self.bin_quantiles = np.linspace(0, 1, self.n_bins + 1)

        super().__init__(model=model)

    def _calibrate_interval(
        self,
        data: Any,
        response: Union[np.ndarray, pd.Series],
        alpha: Union[int, float] = 0.95,
    ) -> None:
        """Method to set the baseline conformal intervals depending on
        the value of the scaling factors.

        First the scaling factors for `data` are calculated then the
        quantiles (defined in the bin_quantiles attribute) of the scaling
        factors are calculated. Next the scaling factors are bucketed
        at these quantiles. Finally the `alpha` quantiles of the scaled
        nonconformity values are calculated for each bin.

        Results are stored in the `baseline_intervals` attribute. The
        edges for the bins are stored in the `scaling_factor_cut_points`
        attribute.

        The alpha value is also stored in an attribute of the same name.

        Parameters
        ----------
        data : Any
            Dataset to use to set baseline interval width.

        response : np.ndarray or pd.Series
            The response values for the records in data.

        alpha : int or float, default = 0.95
            Confidence level for the interval.

        """

        check_attribute(
            self,
            "leaf_node_counts",
            "object does not have leaf_node_counts attribute, run calibrate first.",
        )

        self.alpha = alpha

        predictions = predictions = self._generate_predictions(data)

        scaling_factors = self._calculate_scaling_factors(data)

        nonconformity_values = nonconformity.scaled_absolute_error(
            predictions=predictions, response=response, scaling=scaling_factors
        )

        scaling_factor_cut_points = np.quantile(scaling_factors, self.bin_quantiles)

        self.scaling_factor_cut_points = scaling_factor_cut_points

        # bins will be of the form; bin[i-1] < x <= bin[i]
        # meaning the top bin will have only 1 observation in it,
        # the maximum value in the dataset
        scaling_factor_bins = np.digitize(
            x=scaling_factors, bins=scaling_factor_cut_points, right=True
        )

        n_bins = len(scaling_factor_cut_points) - 1
        self.n_bins = n_bins

        # with right = True specified in np.digitize any values equal
        # to the min will fall into bin 0, so group into bin 1
        scaling_factor_bins = np.clip(scaling_factor_bins, a_min=1, a_max=n_bins)

        baseline_intervals = []

        for bin in range(1, n_bins + 1):

            bin_quantile = nonconformity.nonconformity_at_alpha(
                nonconformity_values[scaling_factor_bins == bin], alpha
            )

            baseline_intervals.append(bin_quantile)

        self.baseline_intervals = np.array(baseline_intervals)

        self._check_interval_monotonicity()

    def predict_with_interval(self, data: Any) -> np.ndarray:
        """Method to generate predictions on data with conformal intervals.

        Each prediction is produced with an associated conformal interval.
        The default interval is of a fixed width and this is scaled
        differently for each row. The default interval also depends on the
        scaling factor value. For each row the selected default/baseline
        interval is then multiplied by the scaling factor.

        The scaling factors are derived by counting the number of times
        each leaf node, visited to make the prediction, was visited in
        predicting the calibration dataset.

        The method is very similar to the
        `LeafNodeScaledConformalPredictor.predict_with_interval` method,
        with the only difference being that the baseline interval is looked up
        from the range of values using the scaling factors for each row.

        Parameters
        ----------
        data : Any
            Data to generate predictions with conformal intervals on.

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
            "baseline_intervals",
            "object does not have baseline_intervals attribute, run calibrate first.",
        )

        predictions = self._generate_predictions(data)

        n_preds = predictions.shape[0]

        scaling_factors = self._calculate_scaling_factors(data)

        baseline_interval = self._lookup_baseline_interval(scaling_factors)

        lower_interval = predictions - (baseline_interval * scaling_factors)
        upper_interval = predictions + (baseline_interval * scaling_factors)

        predictions_with_interval = np.concatenate(
            (
                lower_interval.reshape((n_preds, 1)),
                predictions.reshape((n_preds, 1)),
                upper_interval.reshape((n_preds, 1)),
            ),
            axis=1,
        )

        return predictions_with_interval

    def _lookup_baseline_interval(self, scaling_factors: Any) -> np.ndarray:
        """Lookup the baseline intervals to use given the scaling factor
        values passed.
        """

        bin_index_lookup = np.searchsorted(
            a=self.scaling_factor_cut_points, v=scaling_factors, side="left"
        )

        bin_index_lookup = np.clip(bin_index_lookup, a_min=1, a_max=self.n_bins)

        interval_lookup = self.baseline_intervals[bin_index_lookup - 1]

        return interval_lookup

    def _check_interval_monotonicity(self) -> None:
        """Check that the baseline intervals that have been calculated are either
        monotonically increasing or decreasing.

        A warning is raised if the intervals are not monotonic in either direction.

        """

        monotonically_increasing = np.all(np.diff(self.baseline_intervals) >= 0)

        monotonically_decreasing = np.all(np.diff(self.baseline_intervals) <= 0)

        if not monotonically_increasing and not monotonically_decreasing:

            warnings.warn(
                f"baseline intervals calculated on {self.n_bins} bins are not "
                "monotonic in either direction"
            )
