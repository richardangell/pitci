import abc

from pitci.lightgbm import LGBMBoosterLeafNodeSplitConformalPredictor
import pitci


def test_mro():
    """Test the inheritance order is correct."""

    expected_mro = tuple(
        [
            pitci.lightgbm.LGBMBoosterLeafNodeSplitConformalPredictor,
            pitci.base.SplitConformalPredictor,
            pitci.lightgbm.LGBMBoosterLeafNodeScaledConformalPredictor,
            pitci.base.LeafNodeScaledConformalPredictor,
            abc.ABC,
            object,
        ]
    )

    assert (
        LGBMBoosterLeafNodeSplitConformalPredictor.__mro__ == expected_mro
    ), "mro not correct for LGBMBoosterLeafNodeSplitConformalPredictor"
