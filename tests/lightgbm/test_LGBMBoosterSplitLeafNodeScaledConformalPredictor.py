import abc

from pitci.lightgbm import LGBMBoosterSplitLeafNodeScaledConformalPredictor
import pitci


def test_mro():
    """Test the inheritance order is correct."""

    expected_mro = tuple(
        [
            pitci.lightgbm.LGBMBoosterSplitLeafNodeScaledConformalPredictor,
            pitci.base.SplitConformalPredictorMixin,
            pitci.lightgbm.LGBMBoosterLeafNodeScaledConformalPredictor,
            pitci.base.LeafNodeScaledConformalPredictor,
            pitci.base.ConformalPredictor,
            abc.ABC,
            object,
        ]
    )

    assert (
        LGBMBoosterSplitLeafNodeScaledConformalPredictor.__mro__ == expected_mro
    ), "mro not correct for LGBMBoosterSplitLeafNodeScaledConformalPredictor"


class TestConformalPredictionValues:
    """Baseline tests of the conformal predictions from the
    XGBoosterLeafNodeScaledConformalPredictor class.
    """

    def test_conformal_predictions(
        self, lgbmbooster_diabetes_model, split_diabetes_data_into_4
    ):
        """Test that the conformal intervals are as expected."""

        confo_model = pitci.get_split_leaf_node_scaled_conformal_predictor(
            lgbmbooster_diabetes_model
        )

        confo_model.calibrate(
            data=split_diabetes_data_into_4[6],
            alpha=0.8,
            response=split_diabetes_data_into_4[7],
        )

        assert confo_model.baseline_interval.tolist() == [
            149850.56265605448,
            137606.102224906,
            131990.08520407058,
        ], "baseline_intervals not calculated as expected on diabetes dataset"

        predictions_test = confo_model.predict_with_interval(
            split_diabetes_data_into_4[6]
        )

        assert (
            round(float(predictions_test[:, 1].mean()), 7) == 158.5720143
        ), "mean test sample predicted value not calculated as expected on diabetes dataset"

        expected_interval_distribution = {
            0.0: 177.04907472041657,
            0.05: 178.22323227337145,
            0.1: 178.77408054789797,
            0.2: 179.99190056266366,
            0.3: 180.86654654279693,
            0.4: 182.2483406993801,
            0.5: 183.71976265007476,
            0.6: 184.84256671689613,
            0.7: 187.08785641320517,
            0.8: 189.78042091525793,
            0.9: 192.0194880043173,
            0.95: 192.58427120717985,
            1.0: 193.85583784741846,
            "mean": 184.4546880645055,
            "std": 4.882189594738741,
            "iqr": 7.5686412781202534,
        }

        actual_interval_distribution = pitci.helpers.check_interval_width(
            intervals_with_predictions=predictions_test
        ).to_dict()

        assert (
            expected_interval_distribution == actual_interval_distribution
        ), "conformal interval distribution not calculated as expected"
