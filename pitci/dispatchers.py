"""
Module containing functions used to return the correct conformal predictor
class given the underlying model type.
"""

from functools import singledispatch


@singledispatch
def get_absolute_error_conformal_predictor(model):
    """Function to return the appropriate child class of
    AbsoluteErrorConformalPredictor depending on the type
    of the model arg.
    """

    raise NotImplementedError(
        f"model type not supported for AbsoluteErrorConformalPredictor children; {type(model)}"
    )


@singledispatch
def get_leaf_node_scaled_conformal_predictor(model):
    """Function to return the appropriate child class of
    LeafNodeScaledConformalPredictor depending on the type
    of the model arg.
    """

    raise NotImplementedError(
        f"model type not supported for LeafNodeScaledConformalPredictor children; {type(model)}"
    )


@singledispatch
def get_leaf_node_split_conformal_predictor(model):
    """Function to return the appropriate child class of
    SplitConformalPredictor depending on the type of the model arg.
    """

    raise NotImplementedError(
        f"model type not supported for SplitConformalPredictor children; {type(model)}"
    )
