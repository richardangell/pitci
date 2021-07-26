"""
Module containing base conformal predictor classes, model-type specific conformal
predictor classes will inherit from these base classes.
"""

import pandas as pd
import numpy as np
import warnings
from abc import ABC, abstractmethod

from typing import Union, Any, List, Dict, Optional

from ._version import __version__
from .checks import (
    check_type,
    check_attribute,
)
from . import nonconformity
from . import docstrings


class AbsoluteErrorConformalPredictor(ABC):
    """Conformal interval predictor for an underlying {model_type} model using absolute
    error as the nonconformity measure.

    Class implements inductive conformal intervals where a calibration
    dataset is used to learn the information that is used when generating
    intervals for new instances.

    The predictor outputs fixed width intervals for every new instance,
    there is no interval scaling implemented in this class.

    {description}

    Parameters
    ----------
    model : {model_type}
        Underlying {model_type} model to generate prediction intervals with.

    {parameters}

    Attributes
    ----------
    __version__ : str
        The version of the ``pitci`` package that generated the object.

    model : {model_type}
        The underlying {model_type} model passed in initialising the object.

    baseline_interval : float
        The default or baseline conformal half interval width. Will be applied
        without modification to provide an interval for all new instances. Attribute
        is set when the {calibrate_link} method is run.

    alpha : int or float
        The confidence level of the conformal intervals that will be produced.
        Attribute is set when the {calibrate_link} method is run.

    {attributes}

    """

    __doc__: str

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
        """Calibrate conformal intervals that will be applied to new instances
        when calling ``predict_with_interval``.

        {description}

        Parameters
        ----------
        data : {data_type}
            Dataset to calibrate baselines on.

        response : {response_type}
            The associated response values for every record in ``data``.

        alpha : int or float, default = 0.95
            Confidence level for the intervals.

        """

        check_type(alpha, [int, float], "alpha")
        check_type(response, [np.ndarray, pd.Series], "response")

        if not (alpha >= 0 and alpha <= 1):

            raise ValueError("alpha must be in range [0 ,1]")

        self._calibrate_interval(data=data, alpha=alpha, response=response)

    def predict_with_interval(self, data: Any) -> np.ndarray:
        """Generate predictions with conformal intervals using the underlying
        ``model``.

        {description}

        Parameters
        ----------
        data : {data_type}
            Dataset to generate predictions with intervals on.

        Returns
        -------
        predictions_with_interval : np.ndarray
            Array of predictions with intervals for each row in ``data``.
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
        """Generate predictions with underlying model.

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
        """Set the baseline conformal interval. Result is stored in the
        ``baseline_interval`` attribute.

        The value passed in ``alpha`` is also stored in an attribute of the
        same name.

        Parameters
        ----------
        data : Any
            Dataset to use to set baseline interval width.

        response : np.ndarray or pd.Series
            The response values for the records in ``data``.

        alpha : int or float, default = 0.95
            Confidence level for the intervals.

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
    """Conformal interval predictor for an underlying {model_type} model using
    absolute error scaled by leaf node counts as the nonconformity measure.

    Class implements inductive conformal intervals where a calibration
    dataset is used to learn the information that is used when generating
    intervals for new instances.

    The predictor outputs varying width intervals for every new instance. This
    is done by multiplying the ``baseline_interval`` by a scaling factor that
    depends on the input ``data``. The scaling function uses the reciporcal of
    the number of times that the leaf nodes using in making each prediction
    were visited on the calibration dataset, or when the underlying model was
    trained - see the ``train_data`` argument for the :func:`~{calibrate_method}`
    method.

    The intuition behind this is that for rows that have higher leaf node counts
    from the calibration set - the model will be more 'familiar' with hence
    the interval for these rows should be smaller. The inverse is true for rows
    that have lower leaf node counts from the calibration set.

    {description}

    Parameters
    ----------
    model : {model_type}
        Underlying {model_type} model to generate prediction intervals with.

    {parameters}

    Attributes
    ----------
    __version__ : str
        The version of the ``pitci`` package that generated the object.

    model : {model_type}
        The underlying {model_type} model passed in initialising the object.

    leaf_node_counts : list
        The number of times each leaf node in each tree was visited when
        making predictions on the calibration dataset. Each item in the list
        is a ``dict`` giving a mapping between leaf node index and counts
        for a given tree. The length of the list corresponds to the number
        of trees in ``model``.

    baseline_interval : float
        The default or baseline conformal half interval width. Will be scaled
        for each prediction generated.

    alpha : int or float
        The confidence level of the conformal intervals that will be produced.
        Attribute is set when ``_calibrate_interval`` is called by the
        :func:`~{calibrate_method}` method.

    {attributes}

    """

    leaf_node_counts: list
    __doc__: str

    @abstractmethod
    def __init__(self, model: Any) -> None:

        self.__version__ = __version__
        self.model = model

    def calibrate(
        self,
        data: Any,
        response: Union[np.ndarray, pd.Series],
        alpha: Union[int, float] = 0.95,
        train_data: Optional[Any] = None,
    ) -> None:
        """Calibrate conformal intervals to a given sample of ``data`` at a given
        confidence level, ``alpha``, between 0 and 1.

        This method must be run before :func:`~{predict_with_interval_method}` can be
        used to generate predictions.

        There are 2 items to be calibrated; the leaf node counts stored
        in the ``leaf_node_counts`` attribute and the half interval width
        stored in the ``{baseline_interval_attribute}`` attribute.

        The user has the option to specify the training sample that was used
        to buid the model in the ``train_data`` argument. This is to allow the
        ``leaf_node_counts`` to be calibrated on the same data, as the underlying
        model was built on, rather than a separate calibration set which is what
        will be passed in the ``data`` argument. The default interval width for a given
        ``alpha`` has to be set on a separate sample to what was used to build the model.
        If not, the errors will be smaller than they otherwise would be, on a sample
        the underlying model has not seen before. However for the ``leaf_node_counts``,
        ideally we want counts from the train sample - we're not 'learning' anything
        new here, just recreating stats from when the model was built originally.

        {description}

        Parameters
        ----------
        data : {data_type}
            Dataset to use to set baselines.

        response : {response_type}
            The response values for the records in ``data``.

        alpha : int or float, default = 0.95
            Confidence level for the intervals.

        train_data : {train_data_type}
            Optional dataset that can be passed to set baseline ``leaf_node_counts``
            from, separate to the ``data`` arg used to set ``{baseline_interval_attribute}``
            width.

        """

        check_type(response, [pd.Series, np.ndarray], "response")
        check_type(alpha, [int, float], "alpha")

        if not (alpha >= 0 and alpha <= 1):

            raise ValueError("alpha must be in range [0 ,1]")

        if train_data is None:

            self._calibrate_leaf_node_counts(data=data)

        else:

            self._calibrate_leaf_node_counts(data=train_data)

        self._calibrate_interval(data=data, alpha=alpha, response=response)

    def predict_with_interval(self, data: Any) -> np.ndarray:
        """Generate predictions with conformal intervals for the passed ``data``.

        Each prediction is produced with an associated conformal interval.
        The default interval is of a fixed width (``baseline_interval`` attribute) and
        this is scaled differently for each row. The scaling factors are calculated by
        counting the number of times each leaf node, visited to make the prediction,
        was visited in the calibration dataset - looking up values from the
        ``leaf_node_counts`` list.

        Parameters
        ----------
        data : {data_type}
            Data to generate predictions with conformal intervals on.

        Returns
        -------
        predictions_with_interval : np.ndarray
            Array of predictions with intervals for each row in ``data``.
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

        Result is stored in the ``baseline_interval`` attribute.

        The value passed in ``alpha`` is also stored in an attribute of the
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
        """Calculate the scaling factors for a given dataset.

        First leaf node indexes are generated for the passed data using
        the ``_generate_leaf_node_predictions`` method.

        Then leaf node indexes are passed to
        ``_count_leaf_node_visits_from_calibration`` which, for each row,
        counts the total number of times each leaf node index was visited
        in the calibration dataset.

        1 / leaf node counts are returned from this method so that the scaling
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
        """Count the number of times each leaf node was visited across each
        tree in the calibration dataset.

        The function ``_sum_dict_values`` is applied to each row in
        ``leaf_node_predictions``, passing the ``leaf_node_counts`` attribute
        in the ``counts`` arg.

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
        """Set the baseline leaf node counts on the calibration dataset.

        First the ``_generate_leaf_node_predictions`` method is called to
        get the leaf node indexes that were visted in every tree for
        every row in the passed ``data`` arg.

        Then each column in the output from ``_generate_leaf_node_predictions``
        (representing a single tree in the model) is the tabulated to
        count the number of times each leaf node in the tree was
        visited when making predictions for data.

        Parameters
        ----------
        data : Any
            Data to set baseline leaf node counts.

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
        """Generate predictions with underlying model.

        Parameters
        ----------
        data : Any
            Data to generate predictions on.

        """

        pass

    @abstractmethod
    def _generate_leaf_node_predictions(self, data: Any) -> np.ndarray:
        """Generate leaf node predictions with underlying model.

        Specifically this method should return a 2d array where the (i, j)th
        value is the leaf node index for the jth tree used in generating the
        prediction for the ith row.

        Parameters
        ----------
        data : Any
            Data to generate leaf node predictions on.

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
    """Conformal interval predictor for an underlying {model_type} model using
    absolute error scaled by leaf node counts as the nonconformity measure.
    Intervals are also split into bins based off the scaling factors and
    calibrated separately for each bin.

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

    Intervals are split into bins, using the scaling factors, where each bin
    is calibrated at the required confidence level. This addresses the
    situation where the leaf node scaled conformal predictors are not well
    calibrated on subsets of the data, despite being calibrated at the
    required ``alpha`` confidence level overall.

    {description}

    Parameters
    ----------
    model : {model_type}
        Underlying {model_type} model to generate prediction intervals with.

    n_bins : int
        Number of bins to split data into based on the scaling factors.

    {parameters}

    Attributes
    ----------
    __version__ : str
        The version of the ``pitci`` package that generated the object.

    model : {model_type}
        The underlying {model_type} model passed in initialising the object.

    leaf_node_counts : list
        The number of times each leaf node in each tree was visited when
        making predictions on the calibration dataset. Each item in the list
        is a ``dict`` giving a mapping between leaf node index and counts
        for a given tree. The length of the list corresponds to the number
        of trees in ``model``.

    baseline_intervals : list
        The default or baseline conformal half interval widths that depend
        on the scaling factor values. When making prediction intervals
        the correct interval will be looked up based off the scaling factor
        values, this is then multiplied by the scaling factor.

    alpha : int or float
        The confidence level of the conformal intervals that will be produced.
        Attribute is set when the {calibrate_link} method is run.

    n_bins : int
        Number of bins to split data into based off the scaling factors.

    bin_quantiles : float
        Quantiles of the scaling factor values that will be used to define
        the limits of the bins. Attribute is set when the {calibrate_link}
        method is run.

    {attributes}

    """

    def __init__(self, model: Any, n_bins: int = 3) -> None:

        check_type(n_bins, [int], "n_bins")

        if not n_bins > 1:

            raise ValueError("n_bins should be greater than 1")

        self.n_bins = n_bins
        self.bin_quantiles = np.linspace(0, 1, self.n_bins + 1)

        super().__init__(model=model)

    @docstrings.doc_inherit_kwargs(
        LeafNodeScaledConformalPredictor.calibrate,
        style=docstrings.str_format_merge_style,
        description="The ``baseline_intervals`` are each calibrated to the required ``alpha``\n\t"
        "level on the subsets of the data where the scaling factor values\n\t"
        "fall into the range for that particular bucket.",
        predict_with_interval_method=":func:`~pitci.base.LeafNodeScaledConformalPredictor.predict_with_interval`",
        baseline_interval_attribute="baseline_intervals",
        data_type="Any",
        response_type="np.ndarray or pd.Series",
        train_data_type="Any, default = None",
    )
    def calibrate(
        self,
        data: Any,
        response: Union[np.ndarray, pd.Series],
        alpha: Union[int, float] = 0.95,
        train_data: Optional[Any] = None,
    ) -> None:

        super().calibrate(
            data=data, response=response, alpha=alpha, train_data=train_data
        )

    def _calibrate_interval(
        self,
        data: Any,
        response: Union[np.ndarray, pd.Series],
        alpha: Union[int, float] = 0.95,
    ) -> None:
        """Set the baseline conformal intervals depending on the value of the
        scaling factors.

        First the scaling factors for ``data`` are calculated, then the
        quantiles (defined in the ``bin_quantiles attribute``) of the scaling
        factors are calculated. Next the scaling factors are bucketed
        at these quantiles. Finally the ``alpha`` quantiles of the scaled
        nonconformity values are calculated for each bin.

        Results are stored in the ``baseline_intervals`` attribute. The
        edges for the bins are stored in the ``scaling_factor_cut_points``
        attribute.

        The ``alpha`` value is also stored in an attribute of the same name.

        Parameters
        ----------
        data : Any
            Dataset to use to set baseline interval width.

        response : np.ndarray or pd.Series
            The response values for the records in ``data``ß.

        alpha : int or float, default = 0.95
            Confidence level for the intervals.

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
        """Generate predictions with conformal intervals for the passed ``data``.

        Each prediction is produced with an associated conformal interval.
        The default intervals are of a fixed width (``baseline_intervals`` attribute) and
        this is scaled differently for each row. The scaling factors are calculated by
        counting the number of times each leaf node, visited to make the prediction,
        was visited in the calibration dataset - looking up values from the
        ``leaf_node_counts`` list. For the ``SplitConformalPredictor`` class
        the baseline intervals also depend on the sclaing factors - rather than
        there being one interval as in the ``LeafNodeScaledConformalPredictor``
        class.

        The method is very similar to the :func:`~{predict_with_interval_method}`
        method, with the only difference being that the baseline interval is looked up
        from the possible values using the scaling factors for each row.

        Parameters
        ----------
        data : {data_type}
            Data to generate predictions with conformal intervals on.

        Returns
        -------
        predictions_with_interval : np.ndarray
            Array of predictions with intervals for each row in ``data``.
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

        Parameters
        ----------
        scaling_factors : Any
            The scaling factors to lookup the baseline intervals for.

        Returns
        -------
        interval_lookup : np.ndarray
            Array of baseline intervals for each scaling factor passed.

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
