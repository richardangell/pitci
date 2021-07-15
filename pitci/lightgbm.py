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

from pitci.base import LeafNodeScaledConformalPredictor, SplitConformalPredictor
from pitci.checks import check_type, check_allowed_value
from pitci.dispatchers import (
    get_leaf_node_scaled_conformal_predictor,
    get_leaf_node_split_conformal_predictor,
)


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
    """Conformal interval predictor for an underlying lightgbm model
    using scaled absolute error as the nonconformity measure.

    Class implements inductive conformal intervals where a calibration
    dataset is used to learn the information that is used when generating
    intervals for new instances.

    The predictor outputs varying width intervals for every new instance.
    The scaling function uses the number of times that the leaf nodes were
    visited for each tree in making the prediction, for that row, were
    visited in the training dataset.

    Intuitively, for rows that have higher leaf node counts from the train
    set - the model will be more 'familiar' with hence the interval for
    these rows will be shrunk. The inverse is true for rows that have lower
    leaf node counts from the train set.

    The currently supported lgboost objective functions (given the nonconformity
    measure that is based on absolute error) are defined in the
    SUPPORTED_OBJECTIVES attribute.

    Parameters
    ----------
    model : lgb.Booster
        Model to generate predictions with conformal intervals.

    Attributes
    ----------
    model : lgb.Booster
        Model passed in initialisation of the class.

    leaf_node_counts : list
        Counts of number of times each leaf node in each tree was visited when
        making predictions on the train dataset. Attribute is set when the
        calibrate method is run.

    baseline_interval : float
        Default, baseline conformal interval width. Will be scaled for each
        prediction generated. Attribute is set when the calibrate method is
        run.

    alpha : int or float
        The confidence level of the conformal intervals that will be produced.
        Attribute is set when the calibrate method is run.

    SUPPORTED_OBJECTIVES : list
        Booster supported objectives. If an lgb.Booster object is passed using
        a non-supported objective when initialising the class an an error
        will be raised.

    """

    def __init__(self, model: lgb.Booster) -> None:

        check_type(model, [lgb.basic.Booster], "model")

        super().__init__(model=model)

        self.SUPPORTED_OBJECTIVES = SUPPORTED_OBJECTIVES_ABS_ERROR

        check_objective_supported(model, self.SUPPORTED_OBJECTIVES)

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
    """Conformal interval predictor for an underlying `lgb.Booster`
    model using scaled and split absolute error as the nonconformity measure.

    The predictor outputs varying width intervals for every new instance.
    The scaling function uses the number of times that the leaf nodes were
    visited for each tree in making the prediction, for that row, were
    visited in the calibration dataset.

    Intervals are split into bins, using the scaling factors, where each bin
    is calibrated at the required confidence level. This addresses the
    situation that `LGBMBoosterLeafNodeScaledConformalPredictor` can encounter
    where the intervals are calibrated at the overall level for a given
    dataset but subsets of the data are not well calibrated.

    This class combines the methods implemented in SplitConformalPredictor and
    LGBMBoosterLeafNodeScaledConformalPredictor so nothing else is required to
    be implemented in the child class itself.

    """

    pass


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
    model: lgb.Booster,
) -> LGBMBoosterLeafNodeSplitConformalPredictor:
    """Function to return an instance of LGBMBoosterLeafNodeSplitConformalPredictor
    class the passed lgb.Booster object.
    """

    confo_model = LGBMBoosterLeafNodeSplitConformalPredictor(model=model)

    return confo_model
