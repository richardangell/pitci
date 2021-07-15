"""Conformal predictor classes for XGBoost models."""

import pandas as pd
import numpy as np
import json

try:

    import xgboost as xgb

except ModuleNotFoundError as err:

    raise ImportError(
        "xgboost must be installed to use functionality in pitci.xgboost"
    ) from err

from typing import Union, Optional, List, cast

from pitci.base import (
    AbsoluteErrorConformalPredictor,
    LeafNodeScaledConformalPredictor,
    SplitConformalPredictor,
)
from pitci.checks import check_type, check_allowed_value
from pitci.dispatchers import (
    get_leaf_node_scaled_conformal_predictor,
    get_absolute_error_conformal_predictor,
    get_leaf_node_split_conformal_predictor,
)


def check_objective_supported(
    booster: xgb.Booster, supported_objectives: List[str]
) -> None:
    """Function to check that the booster objective parameter is in the
    supported_objectives list and raise and exception if not.
    """

    check_type(booster, [xgb.Booster], "booster")
    check_type(supported_objectives, [list], "supported_objectives")

    for i, objective in enumerate(supported_objectives):

        check_type(objective, [str], f"supported_objectives[{i}]")

    booster_config = json.loads(booster.save_config())

    booster_objective = booster_config["learner"]["objective"]["name"]

    check_allowed_value(
        booster_objective, supported_objectives, "booster objective not supported"
    )


# currently the only supported learning tasks where the basis of the
# nonconformity measure is absoute error - are regression (reg),
# single-class classification (binary) and count prediction (count)
# however not all loss functions within each task are supported
SUPPORTED_OBJECTIVES_ABS_ERROR = [
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


class XGBoosterAbsoluteErrorConformalPredictor(AbsoluteErrorConformalPredictor):
    """Conformal interval predictor for an underlying `xgb.Booster` model
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

    Parameters
    ----------
    model : xgb.Booster
        Underly model to generate prediction intervals for.

    Attributes
    ----------
    model : xgb.Booster
        Underlying model passed in initialisation of the class.

    baseline_interval : float
        Default, baseline conformal interval width. This is the half
        width interval that will be returned for every instance.

    alpha : int or float
        The confidence level of the conformal intervals that will be produced.
        Attribute is set when the calibrate method is run.

    SUPPORTED_OBJECTIVES : list
        Booster supported objectives. If an xgboost model with a non-supported
        objective is passed when initialising the class object an error will be raised.

    """

    def __init__(self, model: xgb.Booster) -> None:

        super().__init__()

        check_type(model, [xgb.Booster], "booster")

        self.SUPPORTED_OBJECTIVES = SUPPORTED_OBJECTIVES_ABS_ERROR

        check_objective_supported(model, self.SUPPORTED_OBJECTIVES)

        self.model = model

    def calibrate(
        self,
        data: xgb.DMatrix,
        response: Optional[Union[np.ndarray, pd.Series]] = None,
        alpha: Union[int, float] = 0.95,
    ) -> None:
        """Method to calibrate conformal intervals that will be applied
        to new instances when calling predict_with_interval.

        Calls the parent class calibrate method after extracting the
        response from the data argument, if response is not passed
        and data is an xgb.DMatrix object.

        Parameters
        ----------
        data : xgb.DMatrix, np.ndarray or pd.DataFrame
            Dataset to calibrate baselines on.

        alpha : int or float, default = 0.95
            Confidence level for the interval.

        response : np.ndarray, pd.Series or None, default = None
            The associated response values for every record in data.

        """

        check_type(data, [xgb.DMatrix], "data")

        if response is None:

            # only to stop mypy complaining about get_label method
            data = cast(xgb.DMatrix, data)

            response = data.get_label()

        super().calibrate(data=data, alpha=alpha, response=response)

    def _generate_predictions(self, data: xgb.DMatrix) -> np.ndarray:
        """Method to generate predictions from the xgboost model.

        Calls predict method on the model attribute with
        ntree_limit = model's best_iteration + 1.

        Parameters
        ----------
        data : xgb.DMatrix, np.ndarray or pd.DataFrame
            Data to generate predictions on.

        """

        check_type(data, [xgb.DMatrix], "data")

        predictions = self.model.predict(
            data, ntree_limit=self.model.best_iteration + 1
        )

        return predictions


class XGBSklearnAbsoluteErrorConformalPredictor(AbsoluteErrorConformalPredictor):
    """Conformal interval predictor for an underlying `xgb.XGBRegressor` or
    `xgb.XGBClassifier` model using non-scaled absolute error as the
    nonconformity measure.

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

    Parameters
    ----------
    model : xgb.XGBRegressor or xgb.XGBClassifier
        Underly model to generate prediction intervals for.

    Attributes
    ----------
    model : xgb.XGBRegressor or xgb.XGBClassifier
        Underlying model passed in initialisation of the class.

    baseline_interval : float
        Default, baseline conformal interval width. This is the half
        width interval that will be returned for every instance.

    alpha : int or float
        The confidence level of the conformal intervals that will be produced.
        Attribute is set when the calibrate method is run.

    SUPPORTED_OBJECTIVES : list
        Booster supported objectives. If an xgboost model with a non-supported
        objective is passed when initialising the class object an error will be raised.

    """

    def __init__(self, model: Union[xgb.XGBRegressor, xgb.XGBClassifier]) -> None:

        super().__init__()

        check_type(model, [xgb.XGBRegressor, xgb.XGBClassifier], "model")

        self.SUPPORTED_OBJECTIVES = SUPPORTED_OBJECTIVES_ABS_ERROR

        check_objective_supported(model.get_booster(), self.SUPPORTED_OBJECTIVES)

        self.model = model

    def calibrate(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        response: Union[np.ndarray, pd.Series],
        alpha: Union[int, float] = 0.95,
    ) -> None:
        """Method to calibrate conformal intervals that will be applied
        to new instances when calling predict_with_interval.

        Calls the parent class calibrate method after extracting the
        response from the data argument, if response is not passed
        and data is an xgb.DMatrix object.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Dataset to calibrate baselines on.

        alpha : int or float, default = 0.95
            Confidence level for the interval.

        response : np.ndarray, pd.Series or None, default = None
            The associated response values for every record in data.

        """

        check_type(data, [np.ndarray, pd.DataFrame], "data")

        super().calibrate(data=data, alpha=alpha, response=response)

    def _generate_predictions(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Method to generate predictions from the xgboost model.

        Calls predict method on the model attribute with
        ntree_limit = model's best_iteration + 1.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Data to generate predictions on.

        """

        check_type(data, [np.ndarray, pd.DataFrame], "data")

        predictions = self.model.predict(
            data, ntree_limit=self.model.best_iteration + 1
        )

        return predictions


class XGBoosterLeafNodeScaledConformalPredictor(LeafNodeScaledConformalPredictor):
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

    The currently supported xgboost objective functions (given the nonconformity
    measure that is based on absolute error) are defined in the
    SUPPORTED_OBJECTIVES attribute.

    Parameters
    ----------
    model : xgb.Booster
        Model to generate predictions with conformal intervals.

    Attributes
    ----------
    model : xgb.Booster
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
        Booster supported objectives. If an xgb.Booster object is passed using
        a non-supported objective when initialising the class an an error
        will be raised.

    """

    def __init__(self, model: xgb.Booster) -> None:

        check_type(model, [xgb.Booster], "model")

        super().__init__(model=model)

        self.SUPPORTED_OBJECTIVES = SUPPORTED_OBJECTIVES_ABS_ERROR

        check_objective_supported(model, self.SUPPORTED_OBJECTIVES)

    def calibrate(
        self,
        data: xgb.DMatrix,
        response: Optional[Union[np.ndarray, pd.Series]] = None,
        alpha: Union[int, float] = 0.95,
        train_data: Optional[xgb.DMatrix] = None,
    ) -> None:
        """Method to calibrate conformal intervals that will allow
        prediction intervals that vary by row.

        There are 2 things that must be calibrated before making predictions;
        the leaf node counts (_calibrate_leaf_node_counts method) and the
        intervals (_calibrate_interval method).

        The user has the option to specify the training sample that was used
        to buid the model in the train_data argument. This is to allow the
        leaf node counts to be calibrated on the training data, what the underlying
        model saw when it was built originally, rather than a separate calibration
        set which is what will be passed in the data arg. The default interval
        width for a given alpha has to be set on a separate sample to what was
        used to build the model. If not, the errors will be smaller than they
        otherwise would be, on a sample the underlying model has not seen before.
        However for the leaf node counts, ideally we want counts from the train
        sample - we're not 'learning' anything new here, just recreating stats
        from when the model was built originally.

        This method is repeating the functionality in LeafNodeScaledConformalPredictor's
        calibrate method so that we can pass different datasets, if
        required, to _calibrate_leaf_node_counts and _calibrate_leaf_node_counts
        methods.

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

        train_data : xgb.DMatrix or None, default = None
            Optional dataset that can be passed to set baseline leaf node counts from, separate
            to the data used to set baseline interval width. With this the user can pass the
            train sample in the train_data arg and the calibration sample in the data so leaf node
            counts do not have to be calibrated on a separate sample, as the intervals do.

        """

        check_type(data, [xgb.DMatrix], "data")
        check_type(train_data, [xgb.DMatrix, type(None)], "train_data")
        check_type(response, [pd.Series, np.ndarray, type(None)], "response")
        check_type(alpha, [int, float], "alpha")

        if not (alpha >= 0 and alpha <= 1):

            raise ValueError("alpha must be in range [0 ,1]")

        if response is None:

            # only to stop mypy complaining about get_label method
            data = cast(xgb.DMatrix, data)

            response = data.get_label()

        if train_data is None:

            self._calibrate_leaf_node_counts(data=data)

        else:

            self._calibrate_leaf_node_counts(data=train_data)

        self._calibrate_interval(data=data, alpha=alpha, response=response)

    def predict_with_interval(self, data: xgb.DMatrix) -> np.ndarray:
        """Method to generate predictions on data with conformal intervals.

        This method runs the xgb.Booster.predict method twice, once to
        generate predictions and once to produce the leaf node indexes.

        Each prediction is produced with an associated conformal interval.
        The default interval is of a fixed width and this is scaled
        differently for each row. Scaling is done, for a given row, by
        counting the number of times each leaf node, visited to make the
        prediction, was visited in the calibration dataset. The counts of
        leaf node visits in the calibration data are set by the
        _calibrate_leaf_node_counts method.

        The scaling factors, generated by _calculate_scaling_factors, are
        multiploed by the baseline_interval value. The scaled nonconformity
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

        predictions_with_interval = super().predict_with_interval(data=data)

        return predictions_with_interval

    def _generate_predictions(self, data: xgb.DMatrix) -> np.ndarray:
        """Method to generate predictions from the xgboost model.

        Calls predict method on the model attribute with
        ntree_limit = model's best_iteration + 1.

        Parameters
        ----------
        data : xgb.DMatrix
            Data to generate predictions on.

        """

        check_type(data, [xgb.DMatrix], "data")

        predictions = self.model.predict(
            data, ntree_limit=self.model.best_iteration + 1
        )

        return predictions

    def _generate_leaf_node_predictions(self, data: xgb.DMatrix) -> np.ndarray:
        """Method to generate leaf node predictions from the xgboost model.

        Method calls xgb.Booster.predict with pred_leaf = True and
        ntree_limit = model's best_iteration + 1.

        If the output of predict is not a 2d matrix the output is shaped to
        be 2d.

        Parameters
        ----------
        data : xgb.DMatrix
            Data to generate predictions on.

        """

        check_type(data, [xgb.DMatrix], "data")

        # matrix of (nsample, ntrees) with each record giving
        # the leaf node of each sample in each tree
        leaf_node_predictions = self.model.predict(
            data=data, pred_leaf=True, ntree_limit=self.model.best_iteration + 1
        )

        # if the input data is a single column reshape the output to
        # be 2d array rather than 1d
        if len(leaf_node_predictions.shape) == 1:

            leaf_node_predictions = leaf_node_predictions.reshape((data.num_row(), 1))

        return leaf_node_predictions


class XGBSklearnLeafNodeScaledConformalPredictor(LeafNodeScaledConformalPredictor):
    """Conformal interval predictor for an underlying `xgboost.XGBRegressor`
    or `xgboost.XGBClassifier` model using scaled absolute error as the
    nonconformity measure.

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

    The currently supported xgboost objective functions (given the nonconformity
    measure that is based on absolute error) are defined in the
    SUPPORTED_OBJECTIVES attribute.

    Parameters
    ----------
    model : xgb.XGBRegressor or xgb.XGBClassifier
        Model to generate predictions with conformal intervals.

    Attributes
    ----------
    model : xgb.XGBRegressor or xgb.XGBClassifier
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
        Booster supported objectives, if an xgb.XGBRegressor or xgb.XGBClassifier
        with a non-supported objective is passed when initialising an instance
        of the class an error will be raised.

    """

    def __init__(self, model: Union[xgb.XGBRegressor, xgb.XGBClassifier]) -> None:

        check_type(model, [xgb.XGBRegressor, xgb.XGBClassifier], "model")

        super().__init__(model=model)

        self.SUPPORTED_OBJECTIVES = SUPPORTED_OBJECTIVES_ABS_ERROR

        check_objective_supported(model.get_booster(), self.SUPPORTED_OBJECTIVES)

    def calibrate(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        response: Union[np.ndarray, pd.Series],
        alpha: Union[int, float] = 0.95,
        train_data: Union[np.ndarray, pd.DataFrame] = None,
    ) -> None:
        """Method to calibrate conformal intervals that will allow
        prediction intervals that vary by row.

        There are 2 things that must be calibrated before making predictions;
        the leaf node counts (_calibrate_leaf_node_counts method) and the
        intervals (_calibrate_interval method).

        The user has the option to specify the training sample that was used
        to buid the model in the train_data argument. This is to allow the
        leaf node counts to be calibrated on the training data, what the underlying
        model saw when it was built originally, rather than a separate calibration
        set which is what will be passed in the data arg. The default interval
        width for a given alpha has to be set on a separate sample to what was
        used to build the model. If not, the errors will be smaller than they
        otherwise would be, on a sample the underlying model has not seen before.
        However for the leaf node counts, ideally we want counts from the train
        sample - we're not 'learning' anything new here, just recreating stats
        from when the model was built originally.

        This method is repeating the functionality in LeafNodeScaledConformalPredictor's
        calibrate method so that we can pass different datasets, if
        required, to _calibrate_leaf_node_counts and _calibrate_leaf_node_counts
        methods.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Dataset to use to set baselines.

        alpha : int or float, default = 0.95
            Confidence level for the interval.

        response : np.ndarray or pd.Series
            The response values for the records in data.

        train_data : np.ndarray, pd.DataFrame or None, default = None
            Optional dataset that can be passed to set baseline leaf node counts from, separate
            to the data used to set baseline interval width. With this the user can pass the
            train sample in the train_data arg and the calibration sample in the data so leaf node
            counts do not have to be calibrated on a separate sample, as the intervals do.

        """

        check_type(data, [np.ndarray, pd.DataFrame], "data")
        check_type(train_data, [np.ndarray, pd.DataFrame, type(None)], "train_data")
        check_type(response, [pd.Series, np.ndarray], "response")
        check_type(alpha, [int, float], "alpha")

        if not (alpha >= 0 and alpha <= 1):

            raise ValueError("alpha must be in range [0 ,1]")

        if train_data is None:

            super()._calibrate_leaf_node_counts(data=data)

        else:

            super()._calibrate_leaf_node_counts(data=train_data)

        super()._calibrate_interval(data=data, alpha=alpha, response=response)

    def predict_with_interval(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Method to generate predictions on data with conformal intervals.

        This method runs the underlying model's predict method twice, once to
        generate predictions and once to produce the leaf node indexes.

        Each prediction is produced with an associated conformal interval.
        The default interval is of a fixed width and this is scaled
        differently for each row. Scaling is done, for a given row, by
        counting the number of times each leaf node, visited to make the
        prediction, was visited in the calibration dataset. The counts of
        leaf node visits in the calibration data are set by the
        _calibrate_leaf_node_counts method.

        The scaling factors, generated by _calculate_scaling_factors, are
        multiploed by the baseline_interval value. The scaled nonconformity
        function implements the inverse and divides the absolute error
        by the scaling factors.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Data to generate predictions with conformal intervals on.

        Returns
        -------
        predictions_with_interval : np.ndarray
            Array of predictions with intervals for each row in data.
            Output array will have 3 columns where the first is the
            lower interval, second are the predictions and the third
            is the upper interval.

        """

        check_type(data, [np.ndarray, pd.DataFrame], "data")

        predictions_with_interval = super().predict_with_interval(data=data)

        return predictions_with_interval

    def _generate_predictions(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Method to generate predictions from the xgboost model.

        Calls predict method on the model attribute with
        ntree_limit = model's best_iteration + 1.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Data to generate predictions on.

        """

        check_type(data, [np.ndarray, pd.DataFrame], "data")

        predictions = self.model.predict(
            data, ntree_limit=self.model.best_iteration + 1
        )

        return predictions

    def _generate_leaf_node_predictions(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Method to generate leaf node predictions from the xgboost model.

        Method calls the underlying model's apply method with ntree_limit =
        model's best_iteration + 1.

        If the output of predict is not a 2d matrix the output is shaped to
        be 2d.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Data to generate predictions on.

        """

        check_type(data, [np.ndarray, pd.DataFrame], "data")

        # matrix of (nsample, ntrees) with each record giving
        # the leaf node of each sample in each tree
        leaf_node_predictions = self.model.apply(
            X=data, ntree_limit=self.model.best_iteration + 1
        )

        # if the input data is a single column reshape the output to
        # be 2d array rather than 1d
        if len(leaf_node_predictions.shape) == 1:

            leaf_node_predictions = leaf_node_predictions.reshape((data.shape[0], 1))

        return leaf_node_predictions


class XGBoosterLeafNodeSplitConformalPredictor(
    SplitConformalPredictor, XGBoosterLeafNodeScaledConformalPredictor
):
    """Conformal interval predictor for an underlying `xgboost.Booster`
    model using scaled and split absolute error as the nonconformity measure.

    The predictor outputs varying width intervals for every new instance.
    The scaling function uses the number of times that the leaf nodes were
    visited for each tree in making the prediction, for that row, were
    visited in the calibration dataset.

    Intervals are split into bins, using the scaling factors, where each bin
    is calibrated at the required confidence level. This addresses the
    situation that `XGBoosterLeafNodeScaledConformalPredictor` can encounter
    where the intervals are calibrated at the overall level for a given
    dataset but subsets of the data are not well calibrated.

    This class combines the methods implemented in SplitConformalPredictor and
    XGBoosterLeafNodeScaledConformalPredictor so nothing else is required to
    be implemented in the child class itself.

    """

    pass


@get_absolute_error_conformal_predictor.register(xgb.Booster)
def return_xgb_booster_absolute_error_confromal_predictor(
    model: xgb.Booster,
) -> XGBoosterAbsoluteErrorConformalPredictor:
    """Function to return an instance of XGBoosterAbsoluteErrorConformalPredictor
    class the passed xgboost model object.
    """

    confo_model = XGBoosterAbsoluteErrorConformalPredictor(model=model)

    return confo_model


@get_absolute_error_conformal_predictor.register(xgb.XGBRegressor)
@get_absolute_error_conformal_predictor.register(xgb.XGBClassifier)
def return_xgb_sklearn_absolute_error_confromal_predictor(
    model: Union[xgb.XGBRegressor, xgb.XGBClassifier]
) -> XGBSklearnAbsoluteErrorConformalPredictor:
    """Function to return an instance of XGBSklearnAbsoluteErrorConformalPredictor
    class the passed xgboost model object.
    """

    confo_model = XGBSklearnAbsoluteErrorConformalPredictor(model=model)

    return confo_model


@get_leaf_node_scaled_conformal_predictor.register(xgb.Booster)
def return_xgb_booster_leaf_node_scaled_confromal_predictor(
    model: xgb.Booster,
) -> XGBoosterLeafNodeScaledConformalPredictor:
    """Function to return an instance of XGBoosterLeafNodeScaledConformalPredictor
    class the passed xgb.Booster object.
    """

    confo_model = XGBoosterLeafNodeScaledConformalPredictor(model=model)

    return confo_model


@get_leaf_node_scaled_conformal_predictor.register(xgb.XGBRegressor)
@get_leaf_node_scaled_conformal_predictor.register(xgb.XGBClassifier)
def return_xgb_sklearn_leaf_node_scaled_confromal_predictor(
    model: Union[xgb.XGBRegressor, xgb.XGBClassifier]
) -> XGBSklearnLeafNodeScaledConformalPredictor:
    """Function to return an instance of XGBSklearnLeafNodeScaledConformalPredictor
    class the passed xgb.XGBRegressor object.
    """

    confo_model = XGBSklearnLeafNodeScaledConformalPredictor(model=model)

    return confo_model


@get_leaf_node_split_conformal_predictor.register(xgb.Booster)
def return_xgb_booster_leaf_node_split_confromal_predictor(
    model: xgb.Booster,
) -> XGBoosterLeafNodeSplitConformalPredictor:
    """Function to return an instance of XGBoosterLeafNodeSplitConformalPredictor
    class the passed xgb.Booster object.
    """

    confo_model = XGBoosterLeafNodeSplitConformalPredictor(model=model)

    return confo_model
