import numpy as np
import pandas as pd
import xgboost as xgb
import re

from pitci.xgboost import XGBSklearnLeafNodeScaledConformalPredictor
import pitci

import pytest


class TestInit:
    """Tests for the XGBSklearnLeafNodeScaledConformalPredictor._init__ method."""

    def test_inheritance(self):
        """Test that XGBSklearnLeafNodeScaledConformalPredictor inherits from
        LeafNodeScaledConformalPredictor.
        """

        assert (
            XGBSklearnLeafNodeScaledConformalPredictor.__mro__[1]
            is pitci.base.LeafNodeScaledConformalPredictor
        ), (
            "XGBSklearnLeafNodeScaledConformalPredictor does not inherit from "
            "LeafNodeScaledConformalPredictor"
        )

    def test_model_type_exception(self):
        """Test an exception is raised if model is not a xgb.XGBRegressor
        or xgb.XGBClassifier object.
        """

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"model is not in expected types {[xgb.XGBRegressor, xgb.XGBClassifier]}, got {tuple}"
            ),
        ):

            XGBSklearnLeafNodeScaledConformalPredictor((1, 2, 3))

    def test_attributes_set(self, xgb_regressor_1_split_1_tree):
        """Test that SUPPORTED_OBJECTIVES, version and model attributes are set."""

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
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

        XGBSklearnLeafNodeScaledConformalPredictor(xgb_regressor_1_split_1_tree)

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
    """Tests for the XGBSklearnLeafNodeScaledConformalPredictor.calibrate method."""

    def test_data_type_exception(self, xgb_regressor_1_split_1_tree):
        """Test an exception is raised if data is not a np.ndarray or pd.DataFrame object."""

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[np.ndarray, pd.DataFrame]}, got {str}"
            ),
        ):

            confo_model.calibrate(data="abcd", response=np.array([1]))

    def test_train_data_type_exception(
        self, np_2x1_with_label, xgb_regressor_1_split_1_tree
    ):
        """Test an exception is raised if train_data is not a np.ndarray, pd.DataFrame object."""

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"train_data is not in expected types {[np.ndarray, pd.DataFrame, type(None)]}, got {str}"
            ),
        ):

            confo_model.calibrate(
                data=np_2x1_with_label[0], train_data="abcd", response=np.array([1])
            )

    def test_alpha_incorrect_type_error(
        self, np_2x1_with_label, xgb_regressor_1_split_1_tree
    ):
        """Test an exception is raised if alpha is not an int or float."""

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"alpha is not in expected types {[int, float]}, got {str}"
            ),
        ):

            confo_model.calibrate(
                data=np_2x1_with_label[0], alpha="abc", response=np.array([0, 1])
            )

    def test_response_incorrect_type_error(
        self, np_2x1_with_label, xgb_regressor_1_split_1_tree
    ):
        """Test an exception is raised if response is not a pd.Series or np.ndarray."""

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"response is not in expected types {[pd.Series, np.ndarray]}, got {bool}"
            ),
        ):

            confo_model.calibrate(data=np_2x1_with_label[0], alpha=0.5, response=False)

    @pytest.mark.parametrize("alpha", [(-0.0001), (-1), (1.0001), (2), (55)])
    def test_alpha_value_error(
        self, np_2x1_with_label, xgb_regressor_1_split_1_tree, alpha
    ):
        """Test an exception is raised if alpha is below 0 or greater than 1."""

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        with pytest.raises(
            ValueError, match=re.escape("alpha must be in range [0 ,1]")
        ):

            confo_model.calibrate(
                data=np_2x1_with_label[0], alpha=alpha, response=np.array([0, 1])
            )

    def test_calibrate_leaf_node_counts_call_train_data_passed(
        self,
        mocker,
        np_2x1_with_label,
        np_4x2_with_label,
        xgb_regressor_1_split_1_tree,
    ):
        """Test LeafNodeScaledConformalPredictor._calibrate_leaf_node_counts call when
        train_data is passed.
        """

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        mocked = mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor, "_calibrate_leaf_node_counts"
        )

        # mock out _calibrate_interval so it does nothing, prevents error
        mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor, "_calibrate_interval"
        )

        confo_model.calibrate(
            data=np_2x1_with_label[0],
            train_data=np_4x2_with_label[0],
            response=np_2x1_with_label[1],
        )

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to LeafNodeScaledConformalPredictor._calibrate_leaf_node_counts"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            call_pos_args == ()
        ), "positional args incorrect in call to LeafNodeScaledConformalPredictor._calibrate_leaf_node_counts"

        assert list(call_kwargs.keys()) == [
            "data"
        ], "incorrect keyword args in call LeafNodeScaledConformalPredictor._calibrate_leaf_node_counts"

        np.testing.assert_array_equal(call_kwargs["data"], np_4x2_with_label[0])

    def test_calibrate_leaf_node_counts_call_train_data_not_passed(
        self, mocker, np_2x1_with_label, xgb_regressor_1_split_1_tree
    ):
        """Test LeafNodeScaledConformalPredictor._calibrate_leaf_node_counts call when
        train_data is not passed.
        """

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        mocked = mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor, "_calibrate_leaf_node_counts"
        )

        # mock out _calibrate_interval so it does nothing, prevents error
        mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor, "_calibrate_interval"
        )

        confo_model.calibrate(data=np_2x1_with_label[0], response=np_2x1_with_label[1])

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to LeafNodeScaledConformalPredictor._calibrate_leaf_node_counts"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            call_pos_args == ()
        ), "positional args incorrect in call to LeafNodeScaledConformalPredictor._calibrate_leaf_node_counts"

        assert list(call_kwargs.keys()) == [
            "data"
        ], "incorrect keyword args in call LeafNodeScaledConformalPredictor._calibrate_leaf_node_counts"

        np.testing.assert_array_equal(call_kwargs["data"], np_2x1_with_label[0])

    def test_super_calibrate_interval_call(
        self, mocker, np_2x1_with_label, xgb_regressor_1_split_1_tree
    ):
        """Test LeafNodeScaledConformalPredictor._calibrate_interval call."""

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        mocked = mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor, "_calibrate_interval"
        )

        response_array = np.array([4, 5])

        confo_model.calibrate(
            data=np_2x1_with_label[0], alpha=0.5, response=response_array
        )

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to LeafNodeScaledConformalPredictor._calibrate_interval"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            call_pos_args == ()
        ), "positional args incorrect in call to LeafNodeScaledConformalPredictor._calibrate_interval"

        assert (
            call_kwargs["alpha"] == 0.5
        ), "alpha incorrect in call to LeafNodeScaledConformalPredictor._calibrate_interval"

        np.testing.assert_array_equal(call_kwargs["response"], response_array)

        np.testing.assert_array_equal(call_kwargs["data"], np_2x1_with_label[0])


class TestPredictWithInterval:
    """Tests for the XGBSklearnLeafNodeScaledConformalPredictor.predict_with_interval method."""

    def test_data_type_exception(self, np_2x1_with_label, xgb_regressor_1_split_1_tree):
        """Test an exception is raised if data is not a xgb.DMatrix object."""

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        confo_model.calibrate(data=np_2x1_with_label[0], response=np_2x1_with_label[1])

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[np.ndarray, pd.DataFrame]}, got {list}"
            ),
        ):

            confo_model.predict_with_interval([])

    def test_no_leaf_node_counts_attribute_exception(
        self, np_2x1_with_label, xgb_regressor_1_split_1_tree
    ):
        """Test an exception is raised if leaf_node_counts attribute is not present."""

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        assert not hasattr(
            confo_model, "leaf_node_counts"
        ), "XGBSklearnLeafNodeScaledConformalPredictor has leaf_node_counts attribute prior to running calibrate"

        with pytest.raises(
            AttributeError,
            match="XGBSklearnLeafNodeScaledConformalPredictor does not have leaf_node_counts"
            " attribute, run calibrate first.",
        ):

            confo_model.predict_with_interval(np_2x1_with_label[0])

    def test_super_predict_with_interval_call(
        self, mocker, np_2x1_with_label, xgb_regressor_1_split_1_tree
    ):
        """Test that LeafNodeScaledConformalPredictor.predict_with_interval is called and the
        outputs of this are returned from the method.
        """

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        confo_model.calibrate(data=np_2x1_with_label[0], response=np_2x1_with_label[1])

        predict_return_value = np.array([200, 101, 1234])

        mocked = mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor,
            "predict_with_interval",
            return_value=predict_return_value,
        )

        results = confo_model.predict_with_interval(np_2x1_with_label[0])

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

        assert list(call_kwargs.keys()) == [
            "data"
        ], "incorrect keyword args in LeafNodeScaledConformalPredictor.predict_with_interval call"

        np.testing.assert_array_equal(call_kwargs["data"], np_2x1_with_label[0])


class TestGeneratePredictions:
    """Tests for the XGBSklearnLeafNodeScaledConformalPredictor._generate_predictions method."""

    def test_data_type_exception(self, np_2x1_with_label, xgb_regressor_1_split_1_tree):
        """Test an exception is raised if data is not a np.ndarray or pd.DataFrame object."""

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[np.ndarray, pd.DataFrame]}, got {float}"
            ),
        ):

            confo_model._generate_predictions(12345.0)

    def test_predict_call(
        self, mocker, np_2x1_with_label, xgb_regressor_1_split_1_tree
    ):
        """Test that the output from xgb.Booster.predict with ntree_limit = best_iteration + 1
        is returned from the method.
        """

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        confo_model.calibrate(np_2x1_with_label[0], np_2x1_with_label[1])

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

        assert (
            len(call_pos_args) == 1
        ), "incorrect number of positional args in xgb.XGBRegressor.predict call"

        np.testing.assert_array_equal(call_pos_args[0], np_2x1_with_label[0])

        assert call_kwargs == {
            "ntree_limit": xgb_regressor_1_split_1_tree.best_iteration + 1
        }, "positional args incorrect in call to xgb.XGBRegressor.predict"


class TestGenerateLeafNodePredictions:
    """Tests for the XGBSklearnLeafNodeScaledConformalPredictor._generate_leaf_node_predictions
    method.
    """

    def test_data_type_exception(self, np_2x1_with_label, xgb_regressor_1_split_1_tree):
        """Test an exception is raised if data is not a np.ndarray or pd.DataFrame object."""

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[np.ndarray, pd.DataFrame]}, got {list}"
            ),
        ):

            confo_model._generate_leaf_node_predictions([])

    def test_predict_call(
        self, mocker, np_2x1_with_label, xgb_regressor_1_split_1_tree
    ):
        """Test that the output from xgb.XGBRegressor.predict with ntree_limit = best_iteration + 1
        and pred_leaf = True is returned from the method.
        """

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        confo_model.calibrate(np_2x1_with_label[0], np_2x1_with_label[1])

        predict_return_value = np.array([[200, 101], [5, 6]])

        mocked = mocker.patch.object(
            xgb.XGBRegressor, "apply", return_value=predict_return_value
        )

        results = confo_model._generate_leaf_node_predictions(np_2x1_with_label[0])

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to xgb.XGBRegressor.apply"

        np.testing.assert_array_equal(results, predict_return_value)

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            len(call_pos_args) == 0
        ), "incorrect number of positional args in xgb.XGBRegressor.apply call"

        assert (
            len(list(call_kwargs.keys())) == 2
        ), "incorrect number of keyword args in xgb.XGBRegressor.apply call"

        assert (
            "ntree_limit" in call_kwargs.keys() and "X" in call_kwargs.keys()
        ), "incorrect keyword args in xgb.XGBRegressor.apply call"

        np.testing.assert_array_equal(call_kwargs["X"], np_2x1_with_label[0])

        assert (
            call_kwargs["ntree_limit"]
            == xgb_regressor_1_split_1_tree.best_iteration + 1
        ), "ntree_limit keyword arg incorrect in xgb.XGBRegressor.apply call"

    def test_output_2d(self, mocker, np_2x1_with_label, xgb_regressor_1_split_1_tree):
        """Test the array returned from _generate_leaf_node_predictions is a 2d array
        even if the output from predict is 1d.
        """

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        confo_model.calibrate(np_2x1_with_label[0], np_2x1_with_label[1])

        # set the return value from predict to be a 1d array
        predict_return_value = np.array([200, 101])

        mocker.patch.object(
            xgb.XGBRegressor, "apply", return_value=predict_return_value
        )

        results = confo_model._generate_leaf_node_predictions(np_2x1_with_label[0])

        expected_results = predict_return_value.reshape(
            predict_return_value.shape[0], 1
        )

        np.testing.assert_array_equal(results, expected_results)
