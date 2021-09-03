import numpy as np
import pandas as pd
import xgboost as xgb
import re

from pitci.xgboost import XGBSklearnAbsoluteErrorConformalPredictor
import pitci

import pytest


class TestInit:
    """Tests for the XGBSklearnAbsoluteErrorConformalPredictor._init__ method."""

    def test_inheritance(self):
        """Test that XGBSklearnAbsoluteErrorConformalPredictor inherits from
        AbsoluteErrorConformalPredictor.
        """

        assert (
            XGBSklearnAbsoluteErrorConformalPredictor.__mro__[1]
            is pitci.base.AbsoluteErrorConformalPredictor
        ), (
            "XGBSklearnAbsoluteErrorConformalPredictor does not inherit from "
            "AbsoluteErrorConformalPredictor"
        )

    def test_model_type_exception(self):
        """Test an exception is raised if model is not a xgb.XGBRegressor or
        xgb.XGBClassifier object.
        """

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"model is not in expected types {[xgb.XGBRegressor, xgb.XGBClassifier]}, got {tuple}"
            ),
        ):

            XGBSklearnAbsoluteErrorConformalPredictor((1, 2, 3))

    def test_attributes_set(self, xgb_regressor_1_split_1_tree):
        """Test that SUPPORTED_OBJECTIVES, version and model attributes are set."""

        confo_model = XGBSklearnAbsoluteErrorConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        assert (
            confo_model.__version__ == pitci.__version__
        ), "__version__ attribute not set to package version value"

        assert (
            confo_model.model is xgb_regressor_1_split_1_tree
        ), "model attribute not set with the value passed in init"

        assert (
            confo_model.SUPPORTED_OBJECTIVES
            == pitci.xgboost.SUPPORTED_OBJECTIVES_ABSOLUTE_ERROR
        ), "SUPPORTED_OBJECTIVES attribute incorrect"

    def test_check_objective_supported_called(
        self, mocker, xgb_regressor_1_split_1_tree
    ):
        """Test that check_objective_supported is called in init."""

        mocked = mocker.patch.object(pitci.xgboost, "check_objective_supported")

        XGBSklearnAbsoluteErrorConformalPredictor(xgb_regressor_1_split_1_tree)

        assert (
            mocked.call_count == 1
        ), "check_objective_supported not called (once) in init"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert call_pos_args == (
            xgb_regressor_1_split_1_tree.get_booster(),
            pitci.xgboost.SUPPORTED_OBJECTIVES_ABSOLUTE_ERROR,
        ), "positional args in check_objective_supported call not correct"

        assert (
            call_kwargs == {}
        ), "keyword args in check_objective_supported call not correct"


class TestCalibrate:
    """Tests for the XGBSklearnAbsoluteErrorConformalPredictor.calibrate method."""

    def test_data_type_exception(self, xgb_regressor_1_split_1_tree):
        """Test an exception is raised if data is not a np.ndarray or pd.DataFrame object."""

        confo_model = XGBSklearnAbsoluteErrorConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[np.ndarray, pd.DataFrame]}, got {list}"
            ),
        ):

            confo_model.calibrate([], np.array([0, 1]))

    def test_super_calibrate_call(
        self, mocker, np_2x1_with_label, xgb_regressor_1_split_1_tree
    ):
        """Test AbsoluteErrorConformalPredictor.calibrate call when response is passed."""

        confo_model = XGBSklearnAbsoluteErrorConformalPredictor(
            xgb_regressor_1_split_1_tree
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
    """Tests for the XGBSklearnAbsoluteErrorConformalPredictor.predict_with_interval method."""

    def test_data_type_exception(self, np_2x1_with_label, xgb_regressor_1_split_1_tree):
        """Test an exception is raised if data is not a pd.DataFrame or np.ndarray object."""

        confo_model = XGBSklearnAbsoluteErrorConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        confo_model.calibrate(np_2x1_with_label[0], np_2x1_with_label[1], 0.8)

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[np.ndarray, pd.DataFrame]}, got {dict}"
            ),
        ):

            confo_model.predict_with_interval({})

    def test_super_predict_with_interval_result_returned(
        self, mocker, np_2x1_with_label, xgb_regressor_1_split_1_tree
    ):
        """Test that super prediction_with_interval is called and the result is returned from
        the function.
        """

        confo_model = XGBSklearnAbsoluteErrorConformalPredictor(
            xgb_regressor_1_split_1_tree
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

        np.testing.assert_array_equal(call_pos_args[0], np_2x1_with_label[0])


class TestGeneratePredictions:
    """Tests for the XGBSklearnAbsoluteErrorConformalPredictor._generate_predictions method."""

    def test_predict_call(
        self, mocker, np_2x1_with_label, xgb_regressor_1_split_1_tree
    ):
        """Test that the output from xgb.Booster.predict with ntree_limit = best_iteration + 1
        is returned from the method.
        """

        confo_model = XGBSklearnAbsoluteErrorConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        confo_model.calibrate(np_2x1_with_label[0], np_2x1_with_label[1], 0.8)

        predict_return_value = np.array([200, 101])

        mocked = mocker.patch.object(
            xgb.XGBRegressor, "predict", return_value=predict_return_value
        )

        results = confo_model._generate_predictions(np_2x1_with_label[0])

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to xgb.XGBRegressor.predict"

        np.testing.assert_array_equal(results, predict_return_value)

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        np.testing.assert_array_equal(np_2x1_with_label[0], call_pos_args[0])

        assert call_kwargs == {
            "ntree_limit": xgb_regressor_1_split_1_tree.best_iteration + 1
        }, "positional args incorrect in call to AbsoluteErrorConformalPredictor.calibrate"


class TestConformalPredictionValues:
    """Baseline tests of the conformal predictions from the
    XGBSklearnAbsoluteErrorConformalPredictor class.
    """

    def test_conformal_predictions(
        self, xgbregressor_diabetes_model, split_diabetes_data_into_4
    ):
        """Test that the conformal intervals are as expected."""

        confo_model = pitci.get_absolute_error_conformal_predictor(
            xgbregressor_diabetes_model
        )

        confo_model.calibrate(
            data=split_diabetes_data_into_4[6],
            alpha=0.8,
            response=split_diabetes_data_into_4[7],
        )

        assert (
            round(float(confo_model.baseline_interval), 7) == 89.2551117
        ), "baseline_interval not calculated as expected on diabetes dataset"

        predictions_test = confo_model.predict_with_interval(
            split_diabetes_data_into_4[6]
        )

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
