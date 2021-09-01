"""Conformal predictor classes for LightGBM models."""

import numpy as np
import pandas as pd

try:

    import lightgbm as lgb

except ModuleNotFoundError as err:

    raise ImportError(
        "lightgbm must be installed to use functionality in pitci.lightgbm"
    ) from err

from typing import List, Union, Any

from .base import LeafNodeScaledConformalPredictor, SplitConformalPredictor
from .checks import check_type, check_allowed_value
from .dispatchers import (
    get_leaf_node_scaled_conformal_predictor,
    get_leaf_node_split_conformal_predictor,
)
from . import docstrings


def check_objective_supported(
    booster: lgb.Booster, supported_objectives: List[str]
) -> None:
    """Function to check that the booster objective parameter is in the
    supported_objectives list and raise and exception if not.
    """

    check_type(booster, [lgb.basic.Booster], "booster")
    check_type(supported_objectives, [list], "supported_objectives")

    for i, objective in enumerate(supported_objectives):

        check_type(objective, [str], f"supported_objectives[{i}]")

    booster_objective = booster.dump_model()["objective"]

    check_allowed_value(
        booster_objective, supported_objectives, "booster objective not supported"
    )


SUPPORTED_OBJECTIVES_ABS_ERROR = [
    "regression",
    "regression_l1",
    "huber",
    "fair",
    "poisson",
    "quantile",
    "mape",
    "gamma",
    "tweedie",
    "binary",
    # "multiclass",
    # "multiclassova",
    # "cross_entropy",
    # "cross_entropy_lambda",
    # "lambdarank",
    # "rank_xendcg"
]


class LGBMBoosterLeafNodeScaledConformalPredictor(LeafNodeScaledConformalPredictor):
    __doc__ = LeafNodeScaledConformalPredictor.__doc__.format(
        model_type="``lgb.Booster``",
        description="The currently supported lgboost objective functions, "
        "given the nonconformity\n    measure that is based on absolute error, are defined "
        "in the\n    SUPPORTED_OBJECTIVES attribute.",
        parameters="",
        attributes="SUPPORTED_OBJECTIVES : list\n"
        "\tBooster supported objectives. If a model with a non-supported "
        "objective\n\tis passed when initialising the class object an error will be raised.",
        calibrate_method="pitci.lightgbm.LGBMBoosterLeafNodeScaledConformalPredictor.calibrate",
    )

    def __init__(self, model: lgb.Booster) -> None:

        check_type(model, [lgb.basic.Booster], "model")

        super().__init__(model=model)

        self.SUPPORTED_OBJECTIVES = SUPPORTED_OBJECTIVES_ABS_ERROR

        check_objective_supported(model, self.SUPPORTED_OBJECTIVES)

    @docstrings.doc_inherit_kwargs(
        LeafNodeScaledConformalPredictor.calibrate,
        style=docstrings.str_format_merge_style,
        description="",
        predict_with_interval_method="pitci.lightgbm.LGBMBoosterLeafNodeScaledConformalPredictor.predict_with_interval",
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
        description="temp",
        data_type="np.ndarray or pd.DataFrame",
    )
    def predict_with_interval(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:

        check_type(data, [np.ndarray, pd.DataFrame], "data")

        return super().predict_with_interval(data=data)

    def _calibrate_leaf_node_counts(self, data: Any) -> None:
        """Method to get the number of times each leaf node was visited on the training
        dataset.

        LightGBM exposes this information through the the `trees_to_dataframe` method. This
        returns a dataframe with tree stats and the `count` column gives the number of records
        in the training data that fall into this node. See
        https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm.Booster.trees_to_dataframe
        for more info.

        """

        trees_df = self.model.trees_to_dataframe()

        leaf_nodes = trees_df.loc[trees_df["split_feature"].isnull()].copy()

        # strip out the number part of the node index
        leaf_nodes["leaf_index"] = (
            leaf_nodes["node_index"]
            .apply(lambda x: x.split("-")[1].replace("L", ""))
            .astype(int)
        )

        self.leaf_node_counts = []

        for tree_no in np.sort(leaf_nodes["tree_index"].unique()):

            tree_leaf_node_counts = {
                row[1]["leaf_index"]: row[1]["count"]
                for row in leaf_nodes.loc[
                    leaf_nodes["tree_index"] == tree_no, ["leaf_index", "count"]
                ].iterrows()
            }

            self.leaf_node_counts.append(tree_leaf_node_counts)

    def _generate_predictions(
        self, data: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Method to generate predictions from the lgboost model.

        The number of trees to predict with is not specified, defaulting to lightgbm's
        default behaviour for the `num_iteration` argument;
        https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm.Booster.predict.

        Parameters
        ----------
        data : lgb.Dataset
            Data to generate predictions on.

        """

        predictions = self.model.predict(data)

        return predictions

    def _generate_leaf_node_predictions(
        self, data: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Method to generate leaf node predictions from the lgboost model.

        Method calls lgb.Booster.predict with pred_leaf = True. Like the
        _generate_predictions method the number of trees to predict with
        is not specified, defaulting to lightgbm's behaviour.

        If the output of predict is not a 2d matrix the output is shaped to
        be 2d.

        Parameters
        ----------
        data : lgb.Dataset
            Data to generate predictions on.

        """

        # matrix of (nsample, ntrees) with each record giving
        # the leaf node of each sample in each tree
        leaf_node_predictions = self.model.predict(data, pred_leaf=True)

        # if the input data is a single column reshape the output to
        # be 2d array rather than 1d
        if len(leaf_node_predictions.shape) == 1:

            leaf_node_predictions = leaf_node_predictions.reshape(
                (leaf_node_predictions.shape[0], 1)
            )

        return leaf_node_predictions


class LGBMBoosterLeafNodeSplitConformalPredictor(
    SplitConformalPredictor, LGBMBoosterLeafNodeScaledConformalPredictor
):
    __doc__ = SplitConformalPredictor.__doc__.format(
        model_type="``lgb.Booster``",
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
        description="The ``baseline_intervals`` are each calibrated to the required ``alpha``\n\t"
        "level on the subsets of the data where the scaling factor values\n\t"
        "fall into the range for that particular bucket.",
        predict_with_interval_method="pitci.lightgbm.LGBMBoosterLeafNodeScaledConformalPredictor.predict_with_interval",
        baseline_interval_attribute="baseline_intervals",
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

        super().calibrate(
            data=data, response=response, alpha=alpha, train_data=train_data
        )

    @docstrings.doc_inherit_kwargs(
        SplitConformalPredictor.predict_with_interval,
        style=docstrings.str_format_merge_style,
        predict_with_interval_method="pitci.xgboost.XGBoosterLeafNodeScaledConformalPredictor.predict_with_interval",
        data_type="pd.DataFrame of np.ndarray",
    )
    def predict_with_interval(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:

        return super().predict_with_interval(data=data)


@get_leaf_node_scaled_conformal_predictor.register(lgb.basic.Booster)
def return_lgb_booster_leaf_node_scaled_confromal_predictor(
    model: lgb.Booster,
) -> LGBMBoosterLeafNodeScaledConformalPredictor:
    """Function to return an instance of LGBMBoosterLeafNodeScaledConformalPredictor
    class the passed lgb.Booster object.
    """

    confo_model = LGBMBoosterLeafNodeScaledConformalPredictor(model=model)

    return confo_model


@get_leaf_node_split_conformal_predictor.register(lgb.basic.Booster)
def return_lgb_booster_leaf_node_split_confromal_predictor(
    model: lgb.Booster, n_bins: int = 3
) -> LGBMBoosterLeafNodeSplitConformalPredictor:
    """Function to return an instance of LGBMBoosterLeafNodeSplitConformalPredictor
    class the passed lgb.Booster object.
    """

    confo_model = LGBMBoosterLeafNodeSplitConformalPredictor(model=model, n_bins=n_bins)

    return confo_model
