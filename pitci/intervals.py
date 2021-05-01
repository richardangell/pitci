import pandas as pd
import numpy as np
import xgboost as xgb

from typing import Union, Optional, Dict, List

from pitci._version import __version__
from pitci.checks import (
    check_type,
    check_attribute,
    check_objective_supported,
    check_allowed_value,
)
import pitci.nonconformity as nonconformity


class AbsoluteErrorConformalPredictor:
    """Conformal interval predictor for an underlying xgboost model
    using non-scaled absolute error as the nonconformity measure.

    Class implements inductive conformal intervals where a calibration
    dataset is used to learn the information that is used when generating
    intervals for new instances.

    The predictor outputs fixed width intervals for every new instance,
    as no scaling is implemented in this class.

    The currently supported xgboost objective functions for the underlying
    model are;
    - binary:logistic
    - reg:logistic
    - reg:squarederror
    - reg:logistic
    - reg:pseudohubererror
    - reg:gamma
    - reg:tweedie
    - count:poisson
    These are held in the SUPPORTED_OBJECTIVES attribute, see note below
    for reasons for excluding some of the reg and binary objectives.

    Note, whenever xgb.Booster.predict is called ntree_limit is used with
    xgb.Booster.best_iteration + 1.

    Parameters
    ----------
    booster : xgb.Booster
        Underly model to generate prediction intervals for.

    Attributes
    ----------
    booster : xgb.Booster
        Underlying model passed in initialisation of the class.

    baseline_interval : float
        Default, baseline conformal interval width. This is the half
        width interval that will be returned for every instance.

    alpha : int or float
        The confidence level of the conformal intervals that will be produced.
        Attribute is set when the calibrate method is run.

    SUPPORTED_OBJECTIVES : list
        Booster supported objectives, if an xgb.Booster object is passed when
        initialising a AbsoluteErrorConformalPredictor object an error will be raised.

    """

    # currently the only supported learning tasks are regression (reg),
    # single-class classification (binary) and count prediction (count)
    # however not all loss functions within each task is supported
    SUPPORTED_OBJECTIVES = [
        "binary:logistic",
        "reg:logistic",
        # 'binary:logitraw',
        # raw logisitic is not supported because calculating the absolute
        # value of the residuals will not make sense when comparing predictions
        # to 0,1 actuals
        # 'binary:hinge'
        # hinge loss is not supported as the outputs are either 0 or 1
        # calculating the absolute residuals in this case will result in
        # only 0, 1 values which will not give a sensible default interval
        # when selecting a quantile
        "reg:squarederror",
        # 'reg:squaredlogerror',
        # squared log error not supported as have found models produce constant
        # predicts instead of a range of values
        "reg:pseudohubererror",
        "reg:gamma",
        "reg:tweedie",
        "count:poisson",
    ]

    def __init__(self, booster: xgb.Booster) -> None:

        self.__version__ = __version__

        check_type(booster, [xgb.Booster], "booster")
        check_objective_supported(booster, self.SUPPORTED_OBJECTIVES)

        self.booster = booster

    def calibrate(
        self,
        data: xgb.DMatrix,
        alpha: Union[int, float] = 0.95,
        response: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> None:
        """Method to calibrate conformal intervals that will be applied
        to new instances when calling predict_with_interval.

        Method calls _calibrate_interval to set the default (fixed width)
        interval.

        Parameters
        ----------
        data : xgb.DMatrix
            Dataset to use to set baselines.

        alpha : int or float, default = 0.95
            Confidence level for the interval.

        response : np.ndarray, pd.Series or None, default = None
            The response values for the records in data. If passed as
            None then the _calibrate_interval function will attempt to extract
            the response from the data argument with get_label.

        """

        check_type(data, [xgb.DMatrix], "data")
        check_type(alpha, [int, float], "alpha")
        check_type(response, [type(None), pd.Series, np.ndarray], "response")

        if not (alpha >= 0 and alpha <= 1):

            raise ValueError("alpha must be in range [0 ,1]")

        self._calibrate_interval(data=data, alpha=alpha, response=response)

    def predict_with_interval(self, data: xgb.DMatrix) -> np.ndarray:
        """Method to generate predictions on data with conformal intervals.

        This method runs the xgb.Booster.predict method once to
        generate predictions and then puts the half interval calculated
        in _calibrate_interval about the predictions.

        Parameters
        ----------
        data : xgb.DMatrix
            Dataset to generate predictions with conformal intervals.

        Returns
        -------
        predictions_with_interval : np.ndarray
            Array of predictions with intervals for each row in data.
            Output array will have 3 columns where the first is the
            lower interval, second are the predictions and the third
            is the upper interval.

        """

        check_type(data, [xgb.DMatrix], "data")
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

    def _generate_predictions(self, data: xgb.DMatrix) -> np.ndarray:
        """Method to generate predictions from the xgboost model.

        Method calls xgb.Booster.predict with ntree_limit =
        booster.best_iteration + 1.

        Parameters
        ----------
        data : xgb.DMatrix
            Data to generate predictions on.

        """

        predictions = self.booster.predict(
            data=data, ntree_limit=self.booster.best_iteration + 1
        )

        return predictions

    def _calibrate_interval(
        self,
        data: xgb.DMatrix,
        alpha: Union[int, float] = 0.95,
        response: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> None:
        """Method to set the baseline conformal interval.

        Result is stored in the baseline_interval attribute.

        The value passed in alpha is also stored in an attribute of the
        same name.

        Parameters
        ----------
        data : xgb.DMatrix
            Dataset to use to set baseline interval width.

        alpha : int or float, default = 0.95
            Confidence level for the interval.

        response : np.ndarray, pd.Series or None, default = None
            The response values for the records in data. If passed as
            None then the function will attempt to extract the response from
            the data argument with get_label.

        """

        self.alpha = alpha

        if response is None:

            response = data.get_label()

        predictions = predictions = self._generate_predictions(data)

        nonconformity_values = nonconformity.absolute_error(
            predictions=predictions, response=response
        )

        self.baseline_interval = np.quantile(nonconformity_values, alpha)


class LeafNodeScaledConformalPredictor(AbsoluteErrorConformalPredictor):
    """Conformal interval predictor for an underlying xgboost model
    using scaled absolute error as the nonconformity measure.

    Class implements inductive conformal intervals where a calibration
    dataset is used to learn the information that is used when generating
    intervals for new instances.

    The predictor outputs varying width intervals for every new instance.
    The scaling function uses the number of times that the leaf nodes were
    visited for each tree in making the prediction (for that row) were
    visited in the calibration dataset.

    Intuitively, for rows that have higher leaf node counts from the calibration
    set - the model will be more 'familiar' with hence the interval for
    these rows will be shrunk. The inverse is true for rows that have lower
    leaf node counts from the calibration set.

    The currently supported xgboost objective functions are defined in the
    SUPPORTED_OBJECTIVES attribute, inherited from the AbsoluteErrorConformalPredictor
    parent class.

    Note, whenever xgb.Booster.predict is called ntree_limit is used with
    xgb.Booster.best_iteration + 1.

    Parameters
    ----------
    booster : xgb.Booster
        Model to generate predictions with conformal intervals.

    Attributes
    ----------
    booster : xgb.Booster
        Model passed in initialisation of the class.

    leaf_node_counts : list
        Counts of number of times each leaf node in each tree was visited when
        making predictions on the calibration dataset. Attribute is set when the
        calibrate method is run, which calls _calibrate_leaf_node_counts. The
        length of the list corresponds to the number of trees.

    baseline_interval : float
        Default, baseline conformal interval width. Will be scaled for each
        prediction generated. Attribute is set when the calibrate method is
        run, which calls _calibrate_interval.

    alpha : int or float
        The confidence level of the conformal intervals that will be produced.
        Attribute is set when the calibrate method is run, which calls
        _calibrate_interval.

    SUPPORTED_OBJECTIVES : list
        Booster supported objectives, if an xgb.Booster object is passed when
        initialising a LeafNodeScaledConformalPredictor object an error
        will be raised.

    SCALING_FUNCTIONS : list
        List of supported scaling functions. Currently only scaling by
        sum of leaf node counts is supported.

    """

    SCALING_FUNCTIONS = ["leaf_node_counts"]

    def __init__(
        self, booster: xgb.Booster, scaling_function: str = "leaf_node_counts"
    ) -> None:

        check_type(scaling_function, [str], "scaling_function")

        check_allowed_value(
            scaling_function, self.SCALING_FUNCTIONS, "scaling_function value invalid"
        )
        self.scaling_function = scaling_function

        super().__init__(booster=booster)

    def calibrate(
        self,
        data: xgb.DMatrix,
        alpha: Union[int, float] = 0.95,
        response: Optional[Union[np.ndarray, pd.Series]] = None,
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

        response : np.ndarray, pd.Series or None, default = None
            The response values for the records in data. If passed as
            None then the _calibrate_interval function will attempt to extract
            the response from the data argument with get_label.

        """

        check_type(data, [xgb.DMatrix], "data")
        check_type(alpha, [int, float], "alpha")
        check_type(response, [type(None), pd.Series, np.ndarray], "response")

        if not (alpha >= 0 and alpha <= 1):
            raise ValueError("alpha must be in range [0 ,1]")

        self._calibrate_leaf_node_counts(data=data)
        self._calibrate_interval(data=data, alpha=alpha, response=response)

    def predict_with_interval(self, data: xgb.DMatrix) -> np.ndarray:
        """Method to generate predictions on data with conformal interval.

        This method runs the xgb.Booster.predict method twice, once to
        generate predictions and once to produce the leaf node indexes.

        Each prediction is produced with an associated conformal interval.
        The default interval is of a fixed width and this is scaled
        differently for each row. Scaling is done (for a given row) by
        counting the number of times each leaf node, visited to make the
        prediction, was visited in the calibration dataset. The counts of
        leaf node visits in the calibration data are set by the
        _calibrate_leaf_node_counts method.

        Method multiplies the scaling factors, generated by _calculate_scaling_factors
        method, by the baseline_interval value. The scaled nonconformity
        function implements the inverse and divides the absolute error
        by the scaling factors.

        Parameters
        ----------
        data : xgb.DMatrix
            Data to generate predictions with conformal intervals on.

        Returns
        -------
        predictions_with_interval : np.ndarray
            Array of predictions with intervals for each row in data.
            Output array will have 3 columns where the first is the
            lower interval, second are the predictions and the third
            is the upper interval.

        """

        check_type(data, [xgb.DMatrix], "data")
        check_attribute(
            self,
            "baseline_interval",
            "LeafNodeScaledConformalPredictor does not have baseline_interval attribute, "
            "run calibrate first.",
        )
        check_attribute(
            self,
            "leaf_node_counts",
            "LeafNodeScaledConformalPredictor does not have leaf_node_counts attribute, "
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

    def _calculate_scaling_factors(self, data: xgb.DMatrix) -> np.ndarray:
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
        data : xgb.DMatrix
            Data to calculate interval scaling factors for.

        Returns
        -------
        leaf_node_counts : np.ndarray
            Array of same length as input data giving factor for each input row.

        """

        leaf_node_predictions = self._generate_leaf_node_predictions(data)

        # count the number of times each leaf node was visited across the
        # training sample
        leaf_node_counts = self._count_leaf_node_visits_from_calibration(
            leaf_node_predictions=leaf_node_predictions
        )

        # change scaling factor to be the smaller the better
        reciprocal_leaf_node_counts = 1 / leaf_node_counts

        return reciprocal_leaf_node_counts

    def _generate_leaf_node_predictions(self, data: xgb.DMatrix) -> np.ndarray:
        """Method to generate leaf node predictions from the xgboost model.

        Method calls xgb.Booster.predict with pred_leaf = True and
        ntree_limit = booster.best_iteration + 1.

        If the output of predict is not a 2d matrix the output is shaped to
        be 2d.

        Parameters
        ----------
        data : xgb.DMatrix
            Data to generate predictions on.

        """

        # matrix of (nsample, ntrees) with each record giving
        # the leaf node of each sample in each tree
        leaf_node_predictions = self.booster.predict(
            data=data, pred_leaf=True, ntree_limit=self.booster.best_iteration + 1
        )

        # if the input data is a single column reshape the output to
        # be 2d array rather than 1d
        if len(leaf_node_predictions.shape) == 1:

            leaf_node_predictions = leaf_node_predictions.reshape((data.num_row(), 1))

        return leaf_node_predictions

    def _get_tree_tabular_structure(self) -> pd.DataFrame:
        """Method to return the xgboost model in a tabular structure.

        Method simply class the trees_to_dataframe method on the booster
        attribute.
        """

        tabular_model = self.booster.trees_to_dataframe()

        return tabular_model

    def _calibrate_leaf_node_counts(self, data: xgb.DMatrix) -> None:
        """Method to set that baseline leaf node occurences.

        The data passed is scored with the pred_leaf option which returns
        the leaf nodes traversed for each tree for each row. Then the number
        of times each leaf node is visited is summed across rows.

        The results are stored in the leaf_node_counts attribute, which is
        a list of length n, where n is the number of trees in the booster
        attribute. Each value in the list is a dict which gives the number
        of times each leaf node in the nth tree was visited in generating
        predictions for data.

        Parameters
        ----------
        data : xgb.DMatrix
            Data to set baseline counts of leaf nodes.

        """

        # convert the xgboost model to tabular structure in order
        # to identify all leaf nodes in the model
        model_df = self._get_tree_tabular_structure()

        leaf_node_predictions = self._generate_leaf_node_predictions(data)

        leaf_node_predictions_df = pd.DataFrame(leaf_node_predictions)

        self.leaf_node_counts = []

        for tree_no, column in enumerate(leaf_node_predictions_df.columns.values):

            # count the number of times each leaf node is visited in
            # each tree for predictions on data
            self.leaf_node_counts.append(
                leaf_node_predictions_df[tree_no].value_counts().to_dict()
            )

            tree_subset = model_df.loc[
                (model_df["Tree"] == tree_no) & (model_df["Feature"] == "Leaf")
            ]

            if not tree_subset.shape[0] > 0:
                raise ValueError(f"no leaf nodes found in model for tree {tree_no}")

            tree_all_leaf_nodes = tree_subset["Node"].tolist()

            # note, because data can be any dataset
            # it is not guaranteed that every single leaf node
            # will be represented there (that would only be
            # the case if data was the exact same
            # dataset as what the model was trained on)
            # so we need to check all leaf nodes and add them
            # into leaf_node_counts, with a count of 0, if
            # they are not already present
            for leaf_node in tree_all_leaf_nodes:

                if leaf_node not in self.leaf_node_counts[tree_no].keys():

                    self.leaf_node_counts[tree_no][leaf_node] = 0

    def _calibrate_interval(
        self,
        data: xgb.DMatrix,
        alpha: Union[int, float] = 0.95,
        response: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> None:
        """Method to set the baseline conformal interval.

        This is the default interval that will be scaled for differently
        for each row.

        Result is stored in the baseline_interval attribute.

        The value passed in alpha is also stored in an attribute of the
        same name.

        Parameters
        ----------
        data : xgb.DMatrix
            Dataset to use to set baseline interval width.

        alpha : int or float, default = 0.95
            Confidence level for the interval.

        response : np.ndarray, pd.Series or None, default = None
            The response values for the records in data. If passed as
            None then the function will attempt to extract the response from
            the data argument with get_label.

        """

        self.alpha = alpha

        if response is None:

            response = data.get_label()

        predictions = predictions = self._generate_predictions(data)

        scaling_factors = self._calculate_scaling_factors(data)

        nonconformity_values = nonconformity.scaled_absolute_error(
            predictions=predictions, response=response, scaling=scaling_factors
        )

        self.baseline_interval = np.quantile(nonconformity_values, alpha)

    def _count_leaf_node_visits_from_calibration(
        self, leaf_node_predictions: np.ndarray
    ) -> np.ndarray:
        """Function to count the number of times each leaf node
        was visited across each tree in the calibration dataset.

        The function _sum_dict_values is applied to each row in
        leaf_node_predictions, passing the leaf_node_counts attribute
        in the counts arg - this contains the number of times each
        leaf node in each tree was visited in making predictions
        for the calibration dataset (set in _calibrate_leaf_node_counts).

        Parameters
        ----------
        leaf_node_predictions : np.ndarray
            Array output from the xgb.Booster.predict method with
            pred_leaf arg set to True giving the leaf node visited
            for each tree when making predictions for each row.

        """

        leaf_node_counts = np.apply_along_axis(
            _sum_dict_values,
            1,
            leaf_node_predictions,
            counts=self.leaf_node_counts,
        )

        return leaf_node_counts


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

        total += counts[i][value]

    return total


class RandomScaledConformalPredictor(LeafNodeScaledConformalPredictor):
    """Conformal interval predictor for an underlying xgboost model
    using scaled absolute error as the nonconformity measure.

    This class implements a random scaling function used to scale
    the interval for each instance.

    This class is meant for investigation rather than serious use.

    """

    def _calculate_scaling_factors(self, data: xgb.DMatrix) -> np.ndarray:
        """Method to calculate the scaling factors for a given dataset.

        Method simply returns a random value between 0 and 1 for each row.
        Seed is not set in the method.

        Parameters
        ----------
        data : xgb.DMatrix
            Data to calculate interval scaling factors for.

        Returns
        -------
        leaf_node_counts : np.ndarray
            Array of same length as input data giving factor for each input row.

        """

        return np.random.random(data.num_row())
