import abc

from pitci.xgboost import XGBoosterSplitLeafNodeScaledConformalPredictor
import pitci


def test_mro():
    """Test the inheritance order is correct."""

    expected_mro = tuple(
        [
            pitci.xgboost.XGBoosterSplitLeafNodeScaledConformalPredictor,
            pitci.base.SplitConformalPredictorMixin,
            pitci.xgboost.XGBoosterLeafNodeScaledConformalPredictor,
            pitci.base.LeafNodeScaledConformalPredictor,
            pitci.base.ConformalPredictor,
            abc.ABC,
            object,
        ]
    )

    assert (
        XGBoosterSplitLeafNodeScaledConformalPredictor.__mro__ == expected_mro
    ), "mro not correct for XGBoosterSplitLeafNodeScaledConformalPredictor"


class TestConformalPredictionValues:
    """Baseline tests of the conformal predictions from the
    XGBoosterLeafNodeScaledConformalPredictor class.
    """

    def test_conformal_predictions(self, xgbooster_diabetes_model, diabetes_xgb_data):
        """Test that the conformal intervals are as expected."""

        confo_model = pitci.get_split_leaf_node_scaled_conformal_predictor(
            xgbooster_diabetes_model
        )

        confo_model.calibrate(data=diabetes_xgb_data[3], alpha=0.8)

        assert confo_model.baseline_interval.tolist() == [
            52449.67460632324,
            35938.095474243164,
            30433.527114868164,
        ], "baseline_intervals not calculated as expected on diabetes dataset"

        predictions_test = confo_model.predict_with_interval(diabetes_xgb_data[3])

        assert (
            round(float(predictions_test[:, 1].mean()), 7) == 145.7608841
        ), "mean test sample predicted value not calculated as expected on diabetes dataset"

        expected_interval_distribution = {
            0.0: 154.24075310833973,
            0.05: 161.00247370080658,
            0.1: 165.87657841857578,
            0.2: 174.11965960739732,
            0.3: 180.9296751324757,
            0.4: 186.39597083478893,
            0.5: 189.6469418165866,
            0.6: 196.14862368960394,
            0.7: 206.9068895361072,
            0.8: 211.98285764900234,
            0.9: 225.44604332796382,
            0.95: 231.43366627276168,
            1.0: 315.3733379779085,
            "mean": 195.34387263584802,
            "std": 28.76987335271672,
            "iqr": 32.30777296481335,
        }

        actual_interval_distribution = pitci.helpers.check_interval_width(
            intervals_with_predictions=predictions_test
        ).to_dict()

        assert (
            expected_interval_distribution == actual_interval_distribution
        ), "conformal interval distribution not calculated as expected"
