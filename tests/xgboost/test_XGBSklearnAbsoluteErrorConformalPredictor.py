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
            == pitci.xgboost.SUPPORTED_OBJECTIVES_ABS_ERROR
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
            pitci.xgboost.SUPPORTED_OBJECTIVES_ABS_ERROR,
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


class TestGeneratePredictions:
    """Tests for the XGBSklearnAbsoluteErrorConformalPredictor._generate_predictions method."""

    def test_data_type_exception(self, np_2x1_with_label, xgb_regressor_1_split_1_tree):
        """Test an exception is raised if data is not a xgb.DMatrix object."""

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

            confo_model._generate_predictions({})

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
