import numpy as np
import xgboost as xgb
import re

from pitci.xgboost import XGBoosterAbsoluteErrorConformalPredictor
import pitci

import pytest


class TestInit:
    """Tests for the XGBoosterAbsoluteErrorConformalPredictor._init__ method."""

    def test_inheritance(self):
        """Test that XGBoosterAbsoluteErrorConformalPredictor inherits from
        AbsoluteErrorConformalPredictor.
        """

        assert (
            XGBoosterAbsoluteErrorConformalPredictor.__mro__[1]
            is pitci.base.AbsoluteErrorConformalPredictor
        ), (
            "XGBoosterAbsoluteErrorConformalPredictor does not inherit from "
            "AbsoluteErrorConformalPredictor"
        )

    def test_model_type_exception(self):
        """Test an exception is raised if model is not a xgb.Booster object."""

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"booster is not in expected types {[xgb.Booster]}, got {tuple}"
            ),
        ):

            XGBoosterAbsoluteErrorConformalPredictor((1, 2, 3))

    def test_attributes_set(self, xgboost_1_split_1_tree):
        """Test that SUPPORTED_OBJECTIVES, version and model attributes are set."""

        confo_model = XGBoosterAbsoluteErrorConformalPredictor(xgboost_1_split_1_tree)

        assert (
            confo_model.__version__ == pitci.__version__
        ), "__version__ attribute not set to package version value"

        assert (
            confo_model.model is xgboost_1_split_1_tree
        ), "model attribute not set with the value passed in init"

        assert (
            confo_model.SUPPORTED_OBJECTIVES
            == pitci.xgboost.SUPPORTED_OBJECTIVES_ABSOLUTE_ERROR
        ), "SUPPORTED_OBJECTIVES attribute incorrect"

    def test_check_objective_supported_called(self, mocker, xgboost_1_split_1_tree):
        """Test that check_objective_supported is called in init."""

        mocked = mocker.patch.object(pitci.xgboost, "check_objective_supported")

        XGBoosterAbsoluteErrorConformalPredictor(xgboost_1_split_1_tree)

        assert (
            mocked.call_count == 1
        ), "check_objective_supported not called (once) in init"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert call_pos_args == (
            xgboost_1_split_1_tree,
            pitci.xgboost.SUPPORTED_OBJECTIVES_ABSOLUTE_ERROR,
        ), "positional args in check_objective_supported call not correct"

        assert (
            call_kwargs == {}
        ), "keyword args in check_objective_supported call not correct"


class TestCalibrate:
    """Tests for the XGBoosterAbsoluteErrorConformalPredictor.calibrate method."""

    def test_data_type_exception(self, xgboost_1_split_1_tree):
        """Test an exception is raised if data is not a xgb.DMatrix object."""

        confo_model = XGBoosterAbsoluteErrorConformalPredictor(xgboost_1_split_1_tree)

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[xgb.DMatrix]}, got {int}"
            ),
        ):

            confo_model.calibrate(12345)

    def test_super_calibrate_call_response_passed(
        self, mocker, dmatrix_2x1_with_label, xgboost_1_split_1_tree
    ):
        """Test AbsoluteErrorConformalPredictor.calibrate call when response is passed."""

        confo_model = XGBoosterAbsoluteErrorConformalPredictor(xgboost_1_split_1_tree)

        mocked = mocker.patch.object(
            pitci.base.AbsoluteErrorConformalPredictor, "calibrate"
        )

        response_array = np.array([4, 5])

        confo_model.calibrate(
            data=dmatrix_2x1_with_label, alpha=0.5, response=response_array
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

        np.testing.assert_array_equal(call_kwargs["response"], response_array)

        assert (
            call_kwargs["data"] == dmatrix_2x1_with_label
        ), "data incorrect in call to AbsoluteErrorConformalPredictor.calibrate"

    def test_super_calibrate_call_no_response_passed(
        self, mocker, dmatrix_2x1_with_label, xgboost_1_split_1_tree
    ):
        """Test AbsoluteErrorConformalPredictor.calibrate call when no response is passed."""

        confo_model = XGBoosterAbsoluteErrorConformalPredictor(xgboost_1_split_1_tree)

        mocked = mocker.patch.object(
            pitci.base.AbsoluteErrorConformalPredictor, "calibrate"
        )

        confo_model.calibrate(data=dmatrix_2x1_with_label, alpha=0.99)

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
            call_kwargs["alpha"] == 0.99
        ), "alpha incorrect in call to AbsoluteErrorConformalPredictor.calibrate"

        np.testing.assert_array_equal(
            call_kwargs["response"], dmatrix_2x1_with_label.get_label()
        )

        assert (
            call_kwargs["data"] == dmatrix_2x1_with_label
        ), "data incorrect in call to AbsoluteErrorConformalPredictor.calibrate"


class TestPredictWithInterval:
    """Tests for the XGBoosterAbsoluteErrorConformalPredictor.predict_with_interval method."""

    def test_data_type_exception(self, dmatrix_2x1_with_label, xgboost_1_split_1_tree):
        """Test an exception is raised if data is not a xgb.DMatrix object."""

        confo_model = XGBoosterAbsoluteErrorConformalPredictor(xgboost_1_split_1_tree)

        confo_model.calibrate(dmatrix_2x1_with_label)

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[xgb.DMatrix]}, got {list}"
            ),
        ):

            confo_model.predict_with_interval([])

    def test_super_predict_with_interval_result_returned(
        self, mocker, dmatrix_2x1_with_label, xgboost_1_split_1_tree
    ):
        """Test that super prediction_with_interval is called and the result is returned from
        the function.
        """

        confo_model = XGBoosterAbsoluteErrorConformalPredictor(xgboost_1_split_1_tree)

        confo_model.calibrate(dmatrix_2x1_with_label)

        predict_return_value = np.array([123, 456])

        mocked = mocker.patch.object(
            pitci.base.AbsoluteErrorConformalPredictor,
            "predict_with_interval",
            return_value=predict_return_value,
        )

        results = confo_model.predict_with_interval(dmatrix_2x1_with_label)

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to AbsoluteErrorConformalPredictor.predict_with_interval"

        np.testing.assert_array_equal(results, predict_return_value)

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]

        assert call_pos_args == (
            dmatrix_2x1_with_label,
        ), "positional args incorrect in call to xgb.Booster.predict"


class TestGeneratePredictions:
    """Tests for the XGBoosterAbsoluteErrorConformalPredictor._generate_predictions method."""

    def test_data_type_exception(self, dmatrix_2x1_with_label, xgboost_1_split_1_tree):
        """Test an exception is raised if data is not a xgb.DMatrix object."""

        confo_model = XGBoosterAbsoluteErrorConformalPredictor(xgboost_1_split_1_tree)

        confo_model.calibrate(dmatrix_2x1_with_label)

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[xgb.DMatrix]}, got {float}"
            ),
        ):

            confo_model._generate_predictions(12345.0)

    def test_predict_call(self, mocker, dmatrix_2x1_with_label, xgboost_1_split_1_tree):
        """Test that the output from xgb.Booster.predict with ntree_limit = best_iteration + 1
        is returned from the method.
        """

        confo_model = XGBoosterAbsoluteErrorConformalPredictor(xgboost_1_split_1_tree)

        confo_model.calibrate(dmatrix_2x1_with_label)

        predict_return_value = np.array([200, 101])

        mocked = mocker.patch.object(
            xgb.Booster, "predict", return_value=predict_return_value
        )

        results = confo_model._generate_predictions(dmatrix_2x1_with_label)

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to xgb.Booster.predict"

        np.testing.assert_array_equal(results, predict_return_value)

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert call_pos_args == (
            dmatrix_2x1_with_label,
        ), "positional args incorrect in call to xgb.Booster.predict"

        assert call_kwargs == {
            "ntree_limit": xgboost_1_split_1_tree.best_iteration + 1
        }, "keyword args incorrect in call to xgb.Booster.predict"


class TestConformalPredictionValues:
    """Baseline tests of the conformal predictions from the
    XGBoosterAbsoluteErrorConformalPredictor class.
    """

    def test_conformal_predictions(self, xgbooster_diabetes_model, diabetes_xgb_data):
        """Test that the conformal intervals are as expected."""

        confo_model = pitci.get_absolute_error_conformal_predictor(
            xgbooster_diabetes_model
        )

        confo_model.calibrate(data=diabetes_xgb_data[3], alpha=0.8)

        assert (
            round(float(confo_model.baseline_interval), 7) == 89.2551117
        ), "baseline_interval not calculated as expected on diabetes dataset"

        predictions_test = confo_model.predict_with_interval(diabetes_xgb_data[3])

        assert (
            round(float(predictions_test[:, 1].mean()), 7) == 145.7608795
        ), "mean test sample predicted value not calculated as expected on diabetes dataset"

        expected_interval_distribution = {
            0.0: 178.5102081298828,
            0.05: 178.5102081298828,
            0.1: 178.51022338867188,
            0.2: 178.51022338867188,
            0.3: 178.51022338867188,
            0.4: 178.51022338867188,
            0.5: 178.51022338867188,
            0.6: 178.51022338867188,
            0.7: 178.51022338867188,
            0.8: 178.51022338867188,
            0.9: 178.51022644042968,
            0.95: 178.51023864746094,
            1.0: 178.51023864746094,
            "mean": 178.51019287109375,
            "std": 6.4099735936906654e-06,
            "iqr": 0.0,
        }

        actual_interval_distribution = pitci.helpers.check_interval_width(
            intervals_with_predictions=predictions_test
        ).to_dict()

        assert (
            expected_interval_distribution == actual_interval_distribution
        ), "conformal interval distribution not calculated as expected"
