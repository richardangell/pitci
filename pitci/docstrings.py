"""
Module providing funtionality for docstring inheriting solution used in the package.
"""

from . import base


def _format_base_class_docstrings() -> None:
    """Function to format the docstrings for the base conformal
    predictor classes in the base module.

    This stops the string formatting keys showing up in the
    documentation.

    """

    base.AbsoluteErrorConformalPredictor.__doc__ = base.AbsoluteErrorConformalPredictor.__doc__.format(
        model_type="``Any``",
        description="",
        parameters="",
        calibrate_link=":func:`~pitci.base.AbsoluteErrorConformalPredictor.calibrate`",
        attributes="",
    )

    base.LeafNodeScaledConformalPredictor.__doc__ = (
        base.LeafNodeScaledConformalPredictor.__doc__.format(
            model_type="``Any``", description="", parameters="", attributes=""
        )
    )

    base.SplitConformalPredictor.__doc__ = base.SplitConformalPredictor.__doc__.format(
        model_type="``Any``",
        description="",
        parameters="",
        calibrate_link="``calibrate``",
        attributes="",
    )
