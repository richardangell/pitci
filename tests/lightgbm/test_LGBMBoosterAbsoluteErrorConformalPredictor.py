import numpy as np
import pandas as pd
import lightgbm as lgb
import re

from pitci.lightgbm import LGBMBoosterAbsoluteErrorConformalPredictor
import pitci

import pytest


class TestInit:
    """Tests for the LGBMBoosterAbsoluteErrorConformalPredictor._init__ method."""

    def test_inheritance(self):
        """Test that LGBMBoosterAbsoluteErrorConformalPredictor inherits from
        AbsoluteErrorConformalPredictor.
        """

        assert (
            LGBMBoosterAbsoluteErrorConformalPredictor.__mro__[1]
            is pitci.base.AbsoluteErrorConformalPredictor
        ), (
            "LGBMBoosterAbsoluteErrorConformalPredictor does not inherit from "
            "AbsoluteErrorConformalPredictor"
        )

    def test_model_type_exception(self):
        """Test an exception is raised if model is not a lgb.Booster object."""

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"model is not in expected types {[lgb.Booster]}, got {list}"
            ),
        ):

            LGBMBoosterAbsoluteErrorConformalPredictor([1, 2, 3])

    def test_attributes_set(self, lgb_booster_1_split_1_tree):
        """Test that SUPPORTED_OBJECTIVES, version and model attributes are set."""

        confo_model = LGBMBoosterAbsoluteErrorConformalPredictor(
            lgb_booster_1_split_1_tree
        )

        assert (
            confo_model.__version__ == pitci.__version__
        ), "__version__ attribute not set to package version value"

        assert (
            confo_model.model is lgb_booster_1_split_1_tree
        ), "model attribute not set with the value passed in init"

        assert (
            confo_model.SUPPORTED_OBJECTIVES
            == pitci.lightgbm.SUPPORTED_OBJECTIVES_ABSOLUTE_ERROR
        ), "SUPPORTED_OBJECTIVES attribute incorrect"

    def test_check_objective_supported_called(self, mocker, lgb_booster_1_split_1_tree):
        """Test that check_objective_supported is called in init."""

        mocked = mocker.patch.object(pitci.lightgbm, "check_objective_supported")

        LGBMBoosterAbsoluteErrorConformalPredictor(lgb_booster_1_split_1_tree)

        assert (
            mocked.call_count == 1
        ), "check_objective_supported not called (once) in init"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert call_pos_args == (
            lgb_booster_1_split_1_tree,
            pitci.lightgbm.SUPPORTED_OBJECTIVES_ABSOLUTE_ERROR,
        ), "positional args in check_objective_supported call not correct"

        assert (
            call_kwargs == {}
        ), "keyword args in check_objective_supported call not correct"


class TestCalibrate:
    """Tests for the LGBMBoosterAbsoluteErrorConformalPredictor.calibrate method."""

    def test_data_type_exception(self, np_2x1_with_label, lgb_booster_1_split_1_tree):
        """Test an exception is raised if data is not a np.ndarray or pd.DataFrame object."""

        confo_model = LGBMBoosterAbsoluteErrorConformalPredictor(
            lgb_booster_1_split_1_tree
        )

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[np.ndarray, pd.DataFrame]}, got {bool}"
            ),
        ):

            confo_model.calibrate(data=False, response=np_2x1_with_label)

    def test_super_calibrate_call(
        self, mocker, np_2x1_with_label, lgb_booster_1_split_1_tree
    ):
        """Test arguments are passed when calling AbsoluteErrorConformalPredictor.calibrate."""

        confo_model = LGBMBoosterAbsoluteErrorConformalPredictor(
            lgb_booster_1_split_1_tree
        )

        mocked = mocker.patch.object(
            pitci.base.AbsoluteErrorConformalPredictor, "calibrate"
        )

        confo_model.calibrate(
            data=np_2x1_with_label[0], alpha=0.5, response=np_2x1_with_label[1]
        )

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to AbsoluteErrorConformalPredictor.calibrate"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            call_pos_args == ()
        ), "positional args incorrect in call to AbsoluteErrorConformalPredictor.calibrate"

        assert (
            call_kwargs["alpha"] == 0.5
        ), "alpha incorrect in call to AbsoluteErrorConformalPredictor.calibrate"

        np.testing.assert_array_equal(call_kwargs["response"], np_2x1_with_label[1])

        np.testing.assert_array_equal(call_kwargs["data"], np_2x1_with_label[0])


class TestPredictWithInterval:
    """Tests for the LGBMBoosterAbsoluteErrorConformalPredictor.predict_with_interval method."""

    def test_data_type_exception(self, np_2x1_with_label, lgb_booster_1_split_1_tree):
        """Test an exception is raised if data is not a np.ndarray or pd.DataFrame object."""

        confo_model = LGBMBoosterAbsoluteErrorConformalPredictor(
            lgb_booster_1_split_1_tree
        )

        confo_model.calibrate(np_2x1_with_label[0], np_2x1_with_label[1])

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[np.ndarray, pd.DataFrame]}, got {pd.Series}"
            ),
        ):

            confo_model.predict_with_interval(pd.Series([1, 2]))

    def test_super_predict_with_interval_result_returned(
        self, mocker, np_2x1_with_label, lgb_booster_1_split_1_tree
    ):
        """Test that super prediction_with_interval is called and the result is returned from
        the function.
        """

        confo_model = LGBMBoosterAbsoluteErrorConformalPredictor(
            lgb_booster_1_split_1_tree
        )

        confo_model.calibrate(np_2x1_with_label[0], np_2x1_with_label[1])

        predict_return_value = np.array([123, 456])

        mocked = mocker.patch.object(
            pitci.base.AbsoluteErrorConformalPredictor,
            "predict_with_interval",
            return_value=predict_return_value,
        )

        results = confo_model.predict_with_interval(np_2x1_with_label[0])

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to AbsoluteErrorConformalPredictor.predict_with_interval"

        np.testing.assert_array_equal(results, predict_return_value)

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]

        assert (
            len(call_pos_args) == 1
        ), "incorrect number of positional args in AbsoluteErrorConformalPredictor.predict_with_interval call"

        np.testing.assert_array_equal(call_pos_args[0], np_2x1_with_label[0])


class TestGeneratePredictions:
    """Tests for the LGBMBoosterAbsoluteErrorConformalPredictor._generate_predictions method."""

    def test_lgb_booster_predict_call(
        self, mocker, np_2x1_with_label, lgb_booster_1_split_1_tree
    ):
        """Test that lgb.Booster.predict is called and the output is returned
        from _generate_predictions.
        """

        confo_model = LGBMBoosterAbsoluteErrorConformalPredictor(
            lgb_booster_1_split_1_tree
        )

        confo_model.calibrate(np_2x1_with_label[0], np_2x1_with_label[1])

        predict_return_value = np.array([200, 101])

        mocked = mocker.patch.object(
            lgb.basic.Booster, "predict", return_value=predict_return_value
        )

        results = confo_model._generate_predictions(np_2x1_with_label[0])

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to lgb.basic.Booster.predict"

        np.testing.assert_array_equal(results, predict_return_value)

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            len(call_pos_args) == 1
        ), "incorrect number of positional args incorrect in call to lgb.Booster.predict"

        np.testing.assert_array_equal(call_pos_args[0], np_2x1_with_label[0])

        assert (
            call_kwargs == {}
        ), "keyword args incorrect in call to xgb.Booster.predict"


class TestConformalPredictionValues:
    """Baseline tests of the conformal predictions from the
    LGBMBoosterAbsoluteErrorConformalPredictor class.
    """

    @pytest.mark.parametrize(
        "alpha", [(0.1), (0.25), (0.5), (0.7), (0.8), (0.9), (0.95), (0.99)]
    )
    def test_calibration(
        self, alpha, lgbmbooster_diabetes_model, split_diabetes_data_into_4
    ):
        """Test that the correct proportion of response values fall within the intervals, on
        the calibration sample.
        """

        confo_model = pitci.get_absolute_error_conformal_predictor(
            lgbmbooster_diabetes_model
        )

        confo_model.calibrate(
            data=split_diabetes_data_into_4[6],
            alpha=alpha,
            response=split_diabetes_data_into_4[7],
        )

        predictions_test = confo_model.predict_with_interval(
            split_diabetes_data_into_4[6]
        )

        calibration_results = pitci.helpers.check_response_within_interval(
            response=split_diabetes_data_into_4[7],
            intervals_with_predictions=predictions_test,
        )

        assert (
            calibration_results[True] >= alpha
        ), f"{type(confo_model)} not calibrated at {alpha}, got {calibration_results[True]}"

    def test_conformal_predictions(
        self, lgbmbooster_diabetes_model, split_diabetes_data_into_4
    ):
        """Test that the conformal intervals are as expected."""

        confo_model = pitci.get_absolute_error_conformal_predictor(
            lgbmbooster_diabetes_model
        )

        confo_model.calibrate(
            data=split_diabetes_data_into_4[6],
            alpha=0.8,
            response=split_diabetes_data_into_4[7],
        )

        assert (
            round(float(confo_model.baseline_interval), 7) == 91.8598813
        ), "baseline_interval not calculated as expected on diabetes dataset"

        predictions_test = confo_model.predict_with_interval(
            split_diabetes_data_into_4[6]
        )

        assert (
            round(float(predictions_test[:, 1].mean()), 7) == 158.5720143
        ), "mean test sample predicted value not calculated as expected on diabetes dataset"

        expected_interval_distribution = {
            0.0: 183.71976265007473,
            0.05: 183.71976265007473,
            0.1: 183.71976265007473,
            0.2: 183.71976265007473,
            0.3: 183.71976265007476,
            0.4: 183.71976265007476,
            0.5: 183.71976265007476,
            0.6: 183.71976265007476,
            0.7: 183.71976265007476,
            0.8: 183.71976265007476,
            0.9: 183.71976265007478,
            0.95: 183.71976265007478,
            1.0: 183.71976265007478,
            "mean": 183.71976265007476,
            "std": 1.7574477124095504e-14,
            "iqr": 2.842170943040401e-14,
        }

        actual_interval_distribution = pitci.helpers.check_interval_width(
            intervals_with_predictions=predictions_test
        ).to_dict()

        assert (
            expected_interval_distribution == actual_interval_distribution
        ), "conformal interval distribution not calculated as expected"
