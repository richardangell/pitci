import numpy as np
import pandas as pd
import xgboost as xgb
import re

from pitci.xgboost import XGBoosterLeafNodeScaledConformalPredictor
import pitci

import pytest


class TestInit:
    """Tests for the XGBoosterLeafNodeScaledConformalPredictor._init__ method."""

    def test_inheritance(self):
        """Test that XGBoosterLeafNodeScaledConformalPredictor inherits from
        LeafNodeScaledConformalPredictor.
        """

        assert (
            XGBoosterLeafNodeScaledConformalPredictor.__mro__[1]
            is pitci.base.LeafNodeScaledConformalPredictor
        ), (
            "XGBoosterLeafNodeScaledConformalPredictor does not inherit from "
            "LeafNodeScaledConformalPredictor"
        )

    def test_model_type_exception(self):
        """Test an exception is raised if model is not a xgb.Booster object."""

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"booster is not in expected types {[xgb.Booster]}, got {tuple}"
            ),
        ):

            XGBoosterLeafNodeScaledConformalPredictor((1, 2, 3))

    def test_attributes_set(self, xgboost_1_split_1_tree):
        """Test that SUPPORTED_OBJECTIVES, version and model attributes are set."""

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

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

        XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

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
    """Tests for the XGBoosterLeafNodeScaledConformalPredictor.calibrate method."""

    def test_data_type_exception(self, xgboost_1_split_1_tree):
        """Test an exception is raised if data is not a xgb.DMatrix object."""

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[xgb.DMatrix]}, got {str}"
            ),
        ):

            confo_model.calibrate("abcd")

    def test_super_calibrate_call_response_passed(
        self, mocker, dmatrix_2x1_with_label, xgboost_1_split_1_tree
    ):
        """Test XGBoosterLeafNodeScaledConformalPredictor.calibrate call when response is passed."""

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        mocked = mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor, "calibrate"
        )

        response_array = np.array([4, 5])

        confo_model.calibrate(
            data=dmatrix_2x1_with_label, alpha=0.5, response=response_array
        )

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to LeafNodeScaledConformalPredictor.calibrate"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            call_pos_args == ()
        ), "positional args incorrect in call to LeafNodeScaledConformalPredictor.calibrate"

        assert (
            call_kwargs["alpha"] == 0.5
        ), "alpha incorrect in call to LeafNodeScaledConformalPredictor.calibrate"

        np.testing.assert_array_equal(call_kwargs["response"], response_array)

        assert (
            call_kwargs["data"] == dmatrix_2x1_with_label
        ), "data incorrect in call to LeafNodeScaledConformalPredictor.calibrate"

    def test_super_calibrate_call_no_response_passed(
        self, mocker, dmatrix_2x1_with_label, xgboost_1_split_1_tree
    ):
        """Test LeafNodeScaledConformalPredictor.calibrate call when no response is passed."""

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        mocked = mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor, "calibrate"
        )

        confo_model.calibrate(data=dmatrix_2x1_with_label, alpha=0.99)

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to LeafNodeScaledConformalPredictor.calibrate"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            call_pos_args == ()
        ), "positional args incorrect in call to LeafNodeScaledConformalPredictor.calibrate"

        assert (
            call_kwargs["alpha"] == 0.99
        ), "alpha incorrect in call to LeafNodeScaledConformalPredictor.calibrate"

        np.testing.assert_array_equal(
            call_kwargs["response"], dmatrix_2x1_with_label.get_label()
        )

        assert (
            call_kwargs["data"] == dmatrix_2x1_with_label
        ), "data incorrect in call to LeafNodeScaledConformalPredictor.calibrate"


class TestPredictWithInterval:
    """Tests for the XGBoosterLeafNodeScaledConformalPredictor.predict_with_interval method."""

    def test_data_type_exception(self, dmatrix_2x1_with_label, xgboost_1_split_1_tree):
        """Test an exception is raised if data is not a xgb.DMatrix object."""

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        confo_model.calibrate(dmatrix_2x1_with_label)

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[xgb.DMatrix]}, got {pd.DataFrame}"
            ),
        ):

            confo_model.predict_with_interval(pd.DataFrame())

    def test_no_leaf_node_counts_attribute_exception(
        self, dmatrix_2x1_with_label, xgboost_1_split_1_tree
    ):
        """Test an exception is raised if leaf_node_counts attribute is not present."""

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        assert not hasattr(
            confo_model, "leaf_node_counts"
        ), "XGBoosterLeafNodeScaledConformalPredictor has leaf_node_counts attribute prior to running calibrate"

        with pytest.raises(
            AttributeError,
            match="XGBoosterLeafNodeScaledConformalPredictor does not have leaf_node_counts"
            " attribute, run calibrate first.",
        ):

            confo_model.predict_with_interval(dmatrix_2x1_with_label)

    def test_super_predict_with_interval_call(
        self, mocker, dmatrix_2x1_with_label, xgboost_1_split_1_tree
    ):
        """Test that LeafNodeScaledConformalPredictor.predict_with_interval is called and the
        outputs of this are returned from the method.
        """

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        confo_model.calibrate(dmatrix_2x1_with_label)

        predict_return_value = np.array([200, 101, 1234])

        mocked = mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor,
            "predict_with_interval",
            return_value=predict_return_value,
        )

        results = confo_model.predict_with_interval(dmatrix_2x1_with_label)

        # test output of predict_with_interval is the return value of
        # LeafNodeScaledConformalPredictor.predict_with_interval
        np.testing.assert_array_equal(results, predict_return_value)

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to super().predict_with_interval"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            call_pos_args == ()
        ), "positional args incorrect in call to LeafNodeScaledConformalPredictor.predict_with_interval"

        assert call_kwargs == {
            "data": dmatrix_2x1_with_label
        }, "keyword args incorrect in call to LeafNodeScaledConformalPredictor.predict_with_interval"


class TestGeneratePredictions:
    """Tests for the XGBoosterLeafNodeScaledConformalPredictor._generate_predictions method."""

    def test_data_type_exception(self, dmatrix_2x1_with_label, xgboost_1_split_1_tree):
        """Test an exception is raised if data is not a xgb.DMatrix object."""

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

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

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

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
        }, "positional args incorrect in call to xgb.Booster.predict"


class TestGenerateLeafNodePredictions:
    """Tests for the XGBoosterLeafNodeScaledConformalPredictor._generate_leaf_node_predictions
    method.
    """

    def test_data_type_exception(self, dmatrix_2x1_with_label, xgboost_1_split_1_tree):
        """Test an exception is raised if data is not a xgb.DMatrix object."""

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[xgb.DMatrix]}, got {list}"
            ),
        ):

            confo_model._generate_leaf_node_predictions([])

    def test_predict_call(self, mocker, dmatrix_2x1_with_label, xgboost_1_split_1_tree):
        """Test that the output from xgb.Booster.predict with ntree_limit = best_iteration + 1
        and pred_leaf = True is returned from the method.
        """

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        confo_model.calibrate(dmatrix_2x1_with_label)

        predict_return_value = np.array([[200, 101], [5, 6]])

        mocked = mocker.patch.object(
            xgb.Booster, "predict", return_value=predict_return_value
        )

        results = confo_model._generate_leaf_node_predictions(dmatrix_2x1_with_label)

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to xgb.Booster.predict"

        np.testing.assert_array_equal(results, predict_return_value)

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            call_pos_args == ()
        ), "positional args incorrect in call to xgb.Booster.predict"

        assert call_kwargs == {
            "ntree_limit": xgboost_1_split_1_tree.best_iteration + 1,
            "data": dmatrix_2x1_with_label,
            "pred_leaf": True,
        }, "positional args incorrect in call to xgb.Booster.predict"

    def test_output_2d(self, mocker, dmatrix_2x1_with_label, xgboost_1_split_1_tree):
        """Test the array returned from _generate_leaf_node_predictions is a 2d array
        even if the output from predict is 1d.
        """

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        confo_model.calibrate(dmatrix_2x1_with_label)

        # set the return value from predict to be a 1d array
        predict_return_value = np.array([200, 101])

        mocker.patch.object(xgb.Booster, "predict", return_value=predict_return_value)

        results = confo_model._generate_leaf_node_predictions(dmatrix_2x1_with_label)

        expected_results = predict_return_value.reshape(
            predict_return_value.shape[0], 1
        )

        np.testing.assert_array_equal(results, expected_results)
