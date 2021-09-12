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
            == pitci.xgboost.SUPPORTED_OBJECTIVES_ABSOLUTE_ERROR
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
            pitci.xgboost.SUPPORTED_OBJECTIVES_ABSOLUTE_ERROR,
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

    def test_super_calibrate_call(
        self, mocker, np_2x1_with_label, xgb_regressor_1_split_1_tree
    ):
        """Test LeafNodeScaledConformalPredictor.calibrate call."""

        confo_model = XGBSklearnLeafNodeScaledConformalPredictor(
            xgb_regressor_1_split_1_tree
        )

        mocked = mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor, "calibrate"
        )

        response_array = np.array([4, 5])

        confo_model.calibrate(
            data=np_2x1_with_label[0],
            alpha=0.5,
            response=response_array,
            train_data=np_2x1_with_label[1],
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

        np.testing.assert_array_equal(call_kwargs["data"], np_2x1_with_label[0])

        np.testing.assert_array_equal(call_kwargs["train_data"], np_2x1_with_label[1])


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


class TestConformalPredictionValues:
    """Baseline tests of the conformal predictions from the
    XGBSklearnLeafNodeScaledConformalPredictor class.
    """

    @pytest.mark.parametrize(
        "alpha", [(0.1), (0.25), (0.5), (0.7), (0.8), (0.9), (0.95), (0.99)]
    )
    def test_calibration(
        self, alpha, xgbregressor_diabetes_model, split_diabetes_data_into_4
    ):
        """Test that the correct proportion of response values fall within the intervals, on
        the calibration sample.
        """

        confo_model = pitci.get_leaf_node_scaled_conformal_predictor(
            xgbregressor_diabetes_model
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
        self, xgbregressor_diabetes_model, split_diabetes_data_into_4
    ):
        """Test that the conformal intervals are as expected."""

        confo_model = pitci.get_leaf_node_scaled_conformal_predictor(
            xgbregressor_diabetes_model
        )

        confo_model.calibrate(
            data=split_diabetes_data_into_4[6],
            alpha=0.8,
            response=split_diabetes_data_into_4[7],
        )

        assert (
            round(float(confo_model.baseline_interval), 7) == 40748.1420135
        ), "baseline_interval not calculated as expected on diabetes dataset"

        predictions_test = confo_model.predict_with_interval(
            split_diabetes_data_into_4[6]
        )

        assert (
            round(float(predictions_test[:, 1].mean()), 7) == 145.7608841
        ), "mean test sample predicted value not calculated as expected on diabetes dataset"

        expected_interval_distribution = {
            0.0: 140.02797942800623,
            0.05: 145.8006552442658,
            0.1: 151.14593459541626,
            0.2: 158.44710522148077,
            0.3: 165.58360738740058,
            0.4: 188.65287738029468,
            0.5: 201.22539265950525,
            0.6: 211.24333094728453,
            0.7: 220.97846697124837,
            0.8: 253.94202300019322,
            0.9: 301.85483776649556,
            0.95: 309.8718023844092,
            1.0: 422.26053900051613,
            "mean": 212.18189767837418,
            "std": 62.11965233604742,
            "iqr": 74.97402168228058,
        }

        actual_interval_distribution = pitci.helpers.check_interval_width(
            intervals_with_predictions=predictions_test
        ).to_dict()

        assert (
            expected_interval_distribution == actual_interval_distribution
        ), "conformal interval distribution not calculated as expected"
