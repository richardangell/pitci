import pandas as pd
import numpy as np
import xgboost as xgb

from typing import Union, Optional

from pitci.checks import (
    check_type,
    check_objective_supported,
)
from pitci.predictors.base import AbsoluteErrorConformalPredictor


class XGBoostAbsoluteErrorConformalPredictor(AbsoluteErrorConformalPredictor):
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
    # however not all loss functions within each task are supported
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

        super().__init__()

        check_type(booster, [xgb.Booster], "booster")
        check_objective_supported(booster, self.SUPPORTED_OBJECTIVES)

        self.booster = booster

    def _generate_predictions(self, data: xgb.DMatrix) -> np.ndarray:
        """Method to generate predictions from the xgboost model.

        Method calls xgb.Booster.predict with ntree_limit =
        booster.best_iteration + 1.

        Parameters
        ----------
        data : xgb.DMatrix
            Data to generate predictions on.

        """

        check_type(data, [xgb.DMatrix], "data")

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

        check_type(data, [xgb.DMatrix], "data")

        if response is None:

            response = data.get_label()

        super()._calibrate_interval(data=data, alpha=alpha, response=response)
