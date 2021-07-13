import abc

from pitci.xgboost import XGBoosterLeafNodeSplitConformalPredictor
import pitci


def test_mro():
    """Test the inheritance order is correct."""

    expected_mro = tuple(
        [
            pitci.xgboost.XGBoosterLeafNodeSplitConformalPredictor,
            pitci.base.SplitConformalPredictor,
            pitci.xgboost.XGBoosterLeafNodeScaledConformalPredictor,
            pitci.base.LeafNodeScaledConformalPredictor,
            abc.ABC,
            object,
        ]
    )

    assert (
        XGBoosterLeafNodeSplitConformalPredictor.__mro__ == expected_mro
    ), "mro not correct for XGBoosterLeafNodeSplitConformalPredictor"
