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
                f"model is not in expected types {[xgb.Booster]}, got {tuple}"
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
            == pitci.xgboost.SUPPORTED_OBJECTIVES_ABS_ERROR
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
            pitci.xgboost.SUPPORTED_OBJECTIVES_ABS_ERROR,
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
        ), "positional args incorrect in call to AbsoluteErrorConformalPredictor.calibrate"

        assert call_kwargs == {
            "ntree_limit": xgboost_1_split_1_tree.best_iteration + 1
        }, "positional args incorrect in call to AbsoluteErrorConformalPredictor.calibrate"
