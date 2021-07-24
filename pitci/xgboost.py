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

from .base import (
    AbsoluteErrorConformalPredictor,
    LeafNodeScaledConformalPredictor,
    SplitConformalPredictor,
)
from .checks import check_type, check_allowed_value
from .dispatchers import (
    get_leaf_node_scaled_conformal_predictor,
    get_absolute_error_conformal_predictor,
    get_leaf_node_split_conformal_predictor,
)
from . import docstrings


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
    __doc__ = AbsoluteErrorConformalPredictor.__doc__.format(
        model_type="``xgb.Booster``",
        description="The currently supported xgboost objective functions, "
        "given the nonconformity\n    measure that is based on absolute error, are defined "
        "in the\n    SUPPORTED_OBJECTIVES attribute.",
        parameters="",
        calibrate_link=":func:`~pitci.xgboost.XGBoosterAbsoluteErrorConformalPredictor.calibrate`",
        attributes="SUPPORTED_OBJECTIVES : list\n"
        "\tBooster supported objectives. If an ``xgb.Booster`` with a non-supported "
        "objective\n\tis passed when initialising the class object an error will be raised.",
    )

    def __init__(self, model: xgb.Booster) -> None:

        check_type(model, [xgb.Booster], "booster")

        super().__init__(model=model)

        self.SUPPORTED_OBJECTIVES = SUPPORTED_OBJECTIVES_ABS_ERROR

        check_objective_supported(model, self.SUPPORTED_OBJECTIVES)

    @docstrings.doc_inherit_kwargs(
        AbsoluteErrorConformalPredictor.calibrate,
        style=docstrings.str_format_merge_style,
        description="Calls the parent class "
        ":func:`~pitci.base.AbsoluteErrorConformalPredictor.calibrate` "
        "method after extracting the\n\t"
        "response from the data argument, if response is not passed\n\t"
        "and data is an xgb.DMatrix object.",
        data_type="xgb.DMatrix, np.ndarray or pd.DataFrame",
        response_type="np.ndarray, pd.Series or None, default = None",
    )
    def calibrate(
        self,
        data: xgb.DMatrix,
        response: Optional[Union[np.ndarray, pd.Series]] = None,
        alpha: Union[int, float] = 0.95,
    ) -> None:

        check_type(data, [xgb.DMatrix], "data")

        if response is None:

            # only to stop mypy complaining about get_label method
            data = cast(xgb.DMatrix, data)

            response = data.get_label()

        super().calibrate(data=data, alpha=alpha, response=response)

    @docstrings.doc_inherit_kwargs(
        AbsoluteErrorConformalPredictor.predict_with_interval,
        style=docstrings.str_format_merge_style,
        description="",
        data_type="xgb.DMatrix",
    )
    def predict_with_interval(self, data: xgb.DMatrix) -> np.ndarray:

        check_type(data, [xgb.DMatrix], "data")

        return super().predict_with_interval(data)

    def _generate_predictions(self, data: xgb.DMatrix) -> np.ndarray:
        """Generate predictions from the xgboost model.

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


class XGBSklearnAbsoluteErrorConformalPredictor(AbsoluteErrorConformalPredictor):
    __doc__ = AbsoluteErrorConformalPredictor.__doc__.format(
        model_type="``xgb.XGBRegressor`` or ``xgb.XGBClassifier``",
        description="The currently supported xgboost objective functions, "
        "given the nonconformity\n    measure that is based on absolute error, are defined "
        "in the\n    SUPPORTED_OBJECTIVES attribute.",
        parameters="",
        calibrate_link=":func:`~pitci.xgboost.XGBSklearnAbsoluteErrorConformalPredictor.calibrate`",
        attributes="SUPPORTED_OBJECTIVES : list\n"
        "\tBooster supported objectives. If an ``xgb.XGBRegressor`` or ``xgb.XGBClassifier`` "
        "with a non-supported objective\n\tis passed when initialising the class object an "
        "error will be raised.",
    )

    def __init__(self, model: Union[xgb.XGBRegressor, xgb.XGBClassifier]) -> None:

        check_type(model, [xgb.XGBRegressor, xgb.XGBClassifier], "model")

        super().__init__(model=model)

        self.SUPPORTED_OBJECTIVES = SUPPORTED_OBJECTIVES_ABS_ERROR

        check_objective_supported(model.get_booster(), self.SUPPORTED_OBJECTIVES)

    @docstrings.doc_inherit_kwargs(
        AbsoluteErrorConformalPredictor.calibrate,
        style=docstrings.str_format_merge_style,
        description="",
        data_type="np.ndarray or pd.DataFrame",
        response_type="np.ndarray or pd.Series",
    )
    def calibrate(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        response: Union[np.ndarray, pd.Series],
        alpha: Union[int, float] = 0.95,
    ) -> None:

        check_type(data, [np.ndarray, pd.DataFrame], "data")

        super().calibrate(data=data, alpha=alpha, response=response)

    @docstrings.doc_inherit_kwargs(
        AbsoluteErrorConformalPredictor.predict_with_interval,
        style=docstrings.str_format_merge_style,
        description="",
        data_type="np.ndarray or pd.DataFrame",
    )
    def predict_with_interval(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:

        check_type(data, [np.ndarray, pd.DataFrame], "data")

        return super().predict_with_interval(data)

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

        predictions = self.model.predict(
            data, ntree_limit=self.model.best_iteration + 1
        )

        return predictions


class XGBoosterLeafNodeScaledConformalPredictor(LeafNodeScaledConformalPredictor):
    __doc__ = LeafNodeScaledConformalPredictor.__doc__.format(
        model_type="``xgb.Booster``",
        description="The currently supported xgboost objective functions, "
        "given the nonconformity\n    measure that is based on absolute error, are defined "
        "in the\n    SUPPORTED_OBJECTIVES attribute.",
        parameters="",
        attributes="SUPPORTED_OBJECTIVES : list\n"
        "\tBooster supported objectives. If an ``xgb.Booster`` with a non-supported "
        "objective\n\tis passed when initialising the class object an error will be raised.",
    )

    def __init__(self, model: xgb.Booster) -> None:

        check_type(model, [xgb.Booster], "model")

        super().__init__(model=model)

        self.SUPPORTED_OBJECTIVES = SUPPORTED_OBJECTIVES_ABS_ERROR

        check_objective_supported(model, self.SUPPORTED_OBJECTIVES)

    @docstrings.doc_inherit_kwargs(
        LeafNodeScaledConformalPredictor.calibrate,
        style=docstrings.str_format_merge_style,
        description="If ``response`` is not passed then the method will attempt to extract\n\t"
        "the response values from ``data`` using the ``get_label`` method.",
        predict_with_interval_method=":func:`~pitci.xgboost.XGBoosterLeafNodeScaledConformalPredictor.predict_with_interval`",
        baseline_interval_attribute="baseline_interval",
        data_type="xgb.DMatrix",
        response_type="np.ndarray, pd.Series or None, default = None",
        train_data_type="xgb.DMatrix or None, default = None",
    )
    def calibrate(
        self,
        data: xgb.DMatrix,
        response: Optional[Union[np.ndarray, pd.Series]] = None,
        alpha: Union[int, float] = 0.95,
        train_data: Optional[xgb.DMatrix] = None,
    ) -> None:

        check_type(data, [xgb.DMatrix], "data")
        check_type(train_data, [xgb.DMatrix, type(None)], "train_data")

        if response is None:

            # only to stop mypy complaining about get_label method
            data = cast(xgb.DMatrix, data)

            response = data.get_label()

        super().calibrate(
            data=data, response=response, alpha=alpha, train_data=train_data
        )

    @docstrings.doc_inherit_kwargs(
        LeafNodeScaledConformalPredictor.predict_with_interval,
        style=docstrings.str_format_merge_style,
        data_type="xgb.DMatrix",
    )
    def predict_with_interval(self, data: xgb.DMatrix) -> np.ndarray:

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
    __doc__ = LeafNodeScaledConformalPredictor.__doc__.format(
        model_type="``xgb.XGBRegressor`` or ``xgb.XGBClassifier``",
        description="The currently supported xgboost objective functions, "
        "given the nonconformity\n    measure that is based on absolute error, are defined "
        "in the\n    SUPPORTED_OBJECTIVES attribute.",
        parameters="",
        attributes="SUPPORTED_OBJECTIVES : list\n"
        "\tBooster supported objectives. If an ``xgb.XGBRegressor`` or ``xgb.XGBClassifier`` "
        "with a non-supported objective\n\tis passed when initialising the class object an "
        "error will be raised.",
    )

    def __init__(self, model: Union[xgb.XGBRegressor, xgb.XGBClassifier]) -> None:

        check_type(model, [xgb.XGBRegressor, xgb.XGBClassifier], "model")

        super().__init__(model=model)

        self.SUPPORTED_OBJECTIVES = SUPPORTED_OBJECTIVES_ABS_ERROR

        check_objective_supported(model.get_booster(), self.SUPPORTED_OBJECTIVES)

    @docstrings.doc_inherit_kwargs(
        LeafNodeScaledConformalPredictor.calibrate,
        style=docstrings.str_format_merge_style,
        description="",
        predict_with_interval_method=":func:`~pitci.xgboost.XGBSklearnLeafNodeScaledConformalPredictor.predict_with_interval`",
        baseline_interval_attribute="baseline_interval",
        data_type="np.ndarray or pd.DataFrame",
        response_type="np.ndarray or pd.Series",
        train_data_type="np.ndarray, pd.DataFrame or None, default = None",
    )
    def calibrate(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        response: Union[np.ndarray, pd.Series],
        alpha: Union[int, float] = 0.95,
        train_data: Union[np.ndarray, pd.DataFrame] = None,
    ) -> None:

        check_type(data, [np.ndarray, pd.DataFrame], "data")
        check_type(train_data, [np.ndarray, pd.DataFrame, type(None)], "train_data")

        super().calibrate(
            data=data, response=response, alpha=alpha, train_data=train_data
        )

    @docstrings.doc_inherit_kwargs(
        LeafNodeScaledConformalPredictor.predict_with_interval,
        style=docstrings.str_format_merge_style,
        data_type="np.ndarray or pd.DataFrame",
    )
    def predict_with_interval(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:

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
    __doc__ = SplitConformalPredictor.__doc__.format(
        model_type="``xgb.Booster``",
        description="The currently supported lgboost objective functions, "
        "given the nonconformity\n    measure that is based on absolute error, are defined "
        "in the\n    SUPPORTED_OBJECTIVES attribute.",
        parameters="",
        calibrate_link="``calibrate``",
        attributes="SUPPORTED_OBJECTIVES : list\n"
        "\tBooster supported objectives. If a model with a non-supported "
        "objective\n\tis passed when initialising the class object an error will be raised.",
    )

    @docstrings.doc_inherit_kwargs(
        LeafNodeScaledConformalPredictor.calibrate,
        style=docstrings.str_format_merge_style,
        description="If ``response`` is not passed then the method will attempt to extract\n\t"
        "the response values from ``data`` using the ``get_label`` method.\n\n\t"
        "The ``baseline_intervals`` are each calibrated to the required ``alpha``\n\t"
        "level on the subsets of the data where the scaling factor values\n\t"
        "fall into the range for that particular bucket.",
        predict_with_interval_method=":func:`~pitci.xgboost.XGBoosterLeafNodeScaledConformalPredictor.predict_with_interval`",
        baseline_interval_attribute="baseline_intervals",
        data_type="xgb.DMatrix",
        response_type="np.ndarray, pd.Series or None, default = None",
        train_data_type="xgb.DMatrix or None, default = None",
    )
    def calibrate(
        self,
        data: xgb.DMatrix,
        response: Optional[Union[np.ndarray, pd.Series]] = None,
        alpha: Union[int, float] = 0.95,
        train_data: Optional[xgb.DMatrix] = None,
    ) -> None:

        super().calibrate(
            data=data, response=response, alpha=alpha, train_data=train_data
        )

    @docstrings.doc_inherit_kwargs(
        SplitConformalPredictor.predict_with_interval,
        style=docstrings.str_format_merge_style,
        predict_with_interval_method="pitci.xgboost.XGBoosterLeafNodeScaledConformalPredictor.predict_with_interval",
        data_type="xgb.DMatrix",
    )
    def predict_with_interval(self, data: xgb.DMatrix) -> np.ndarray:

        super().predict_with_interval(data=data)


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
