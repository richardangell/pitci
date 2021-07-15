import numpy as np
import pandas as pd
import xgboost as xgb
import re

from pitci.base import AbsoluteErrorConformalPredictor
import pitci

import pytest


class DummyAbsoluteErrorConformalPredictor(AbsoluteErrorConformalPredictor):
    """Dummy class inheriting from AbsoluteErrorConformalPredictor so it's
    functionality can be tested.
    """

    def __init__(self, model="abc"):
        """Dummy init method that only calls AbsoluteErrorConformalPredictor
        init method.
        """

        super().__init__(model=model)

    def _generate_predictions(self, data):
        """Dummy function that returns 0s of shape (n,) where data has n rows."""

        return np.zeros(data.shape[0])


class TestInit:
    """Tests for the AbsoluteErrorConformalPredictor._init__ method."""

    def test_version_attribute_set(self):
        """Test that the version attribute is set in init."""

        dummy_confo_model = DummyAbsoluteErrorConformalPredictor()

        assert (
            dummy_confo_model.__version__ == pitci.__version__
        ), "version attribute not set correctly"

    def test_model_attribute_set(self):
        """Test that the model attribute is set in init."""

        dummy_confo_model = DummyAbsoluteErrorConformalPredictor(model=456)

        assert dummy_confo_model.model == 456, "model attribute not set correctly"


class TestCalibrate:
    """Tests for the AbsoluteErrorConformalPredictor.calibrate method."""

    @pytest.mark.parametrize("alpha", [(-0.0001), (-1), (1.0001), (2), (55)])
    def test_alpha_value_error(self, dmatrix_2x1_with_label, alpha):
        """Test an exception is raised if alpha is below 0 or greater than 1."""

        dummy_confo_model = DummyAbsoluteErrorConformalPredictor()

        with pytest.raises(
            ValueError, match=re.escape("alpha must be in range [0 ,1]")
        ):

            dummy_confo_model.calibrate(
                data=dmatrix_2x1_with_label, alpha=alpha, response=np.array([0, 1])
            )

    def test_alpha_incorrect_type_error(self, dmatrix_2x1_with_label):
        """Test an exception is raised if alpha is not an int or float."""

        dummy_confo_model = DummyAbsoluteErrorConformalPredictor()

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"alpha is not in expected types {[int, float]}, got {str}"
            ),
        ):

            dummy_confo_model.calibrate(
                data=dmatrix_2x1_with_label, alpha="abc", response=np.array([0, 1])
            )

    def test_response_incorrect_type_error(self, dmatrix_2x1_with_label):
        """Test an exception is raised if response is not a pd.Series or np.ndarray."""

        dummy_confo_model = DummyAbsoluteErrorConformalPredictor()

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"response is not in expected types {[np.ndarray, pd.Series]}, got {bool}"
            ),
        ):

            dummy_confo_model.calibrate(
                data=dmatrix_2x1_with_label, alpha=0.5, response=False
            )

    def test_calibrate_interval_call(self, mocker, dmatrix_2x1_with_label):
        """Test the call to the _calibrate_interval method."""

        dummy_confo_model = DummyAbsoluteErrorConformalPredictor()

        mocked = mocker.patch.object(
            pitci.base.AbsoluteErrorConformalPredictor, "_calibrate_interval"
        )

        dummy_confo_model.calibrate(
            data=dmatrix_2x1_with_label,
            alpha=0.1,
            response=dmatrix_2x1_with_label.get_label(),
        )

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to _calibrate_interval"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            call_pos_args == ()
        ), "positional args incorrect in _calibrate_interval call"

        np.testing.assert_array_equal(
            call_kwargs["response"], dmatrix_2x1_with_label.get_label()
        )

        assert (
            call_kwargs["alpha"] == 0.1
        ), "alpha arg incorrect in _calibrate_interval call"

        assert (
            call_kwargs["data"] == dmatrix_2x1_with_label
        ), "data arg incorrect in _calibrate_interval call"


class TestPredictWithInterval:
    """Tests for the AbsoluteErrorConformalPredictor.predict_with_interval method."""

    def test_exception_no_baseline_interval(self):
        """Test an exception is raised if no baseline_interval atttibute is present."""

        dummy_confo_model = DummyAbsoluteErrorConformalPredictor()

        assert not hasattr(dummy_confo_model, "baseline_interval")

        with pytest.raises(
            AttributeError,
            match="AbsoluteErrorConformalPredictor does not have baseline_interval attribute, "
            "run calibrate first.",
        ):

            dummy_confo_model.predict_with_interval(np.array([1, 0]))

    def test_expected_output(self):
        """Test the intervals returned are as expected."""

        dummy_confo_model = DummyAbsoluteErrorConformalPredictor()

        dummy_confo_model.baseline_interval = 5

        results = dummy_confo_model.predict_with_interval(np.array([1, 0, -1]))

        expected_results = np.array([[-5, 0, 5], [-5, 0, 5], [-5, 0, 5]])

        np.testing.assert_array_equal(results, expected_results)

    def test_expected_output2(self, mocker):
        """Test the intervals returned are as expected."""

        dummy_confo_model = DummyAbsoluteErrorConformalPredictor()

        dummy_confo_model.baseline_interval = 3

        #  set return value from _generate_predictions
        mocker.patch.object(
            DummyAbsoluteErrorConformalPredictor,
            "_generate_predictions",
            return_value=np.array([1, 2, 3, 4]),
        )

        results = dummy_confo_model.predict_with_interval(np.array([1, 0, -1]))

        expected_results = np.array([[-2, 1, 4], [-1, 2, 5], [0, 3, 6], [1, 4, 7]])

        np.testing.assert_array_equal(results, expected_results)


class TestCalibrateInterval:
    """Tests for the AbsoluteErrorConformalPredictor._calibrate_interval method."""

    def test_alpha_attribute_set(self, dmatrix_2x1_with_label, xgboost_1_split_1_tree):
        """Test that the alpha attribute is set with the passed value."""

        dummy_confo_model = DummyAbsoluteErrorConformalPredictor()

        alpha_value = 0.789

        assert not hasattr(
            dummy_confo_model, "alpha"
        ), "confo model already has alpha attribute"

        dummy_confo_model._calibrate_interval(
            data=np.array([0, 1]), alpha=alpha_value, response=np.array([0, 1])
        )

        assert (
            dummy_confo_model.alpha == alpha_value
        ), "alpha attribute not set to expected value"

    @pytest.mark.parametrize(
        "response, predictions, quantile, expected_baseline_interval",
        [
            (np.array([1, 1, 1]), np.array([1, 2, 3]), 1, 2),
            (np.array([1, 1, 1]), np.array([1, 2, -1]), 1, 2),
            (np.array([1, 1, 1]), np.array([1, 2, -1]), 0.5, 1),
            (np.array([1, 1, 1]), np.array([1, 2, -1]), 0, 0),
            (
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                0.9,
                9,
            ),
        ],
    )
    def test_baseline_interval_expected_value(
        self,
        mocker,
        response,
        predictions,
        quantile,
        expected_baseline_interval,
    ):
        """Test that baseline_interval is calculated correctly."""

        dummy_confo_model = DummyAbsoluteErrorConformalPredictor()

        xgb_dataset = xgb.DMatrix(data=np.ones((response.shape[0], 1)))

        mocker.patch.object(
            DummyAbsoluteErrorConformalPredictor,
            "_generate_predictions",
            return_value=predictions,
        )

        dummy_confo_model._calibrate_interval(
            data=xgb_dataset, alpha=quantile, response=response
        )

        assert (
            dummy_confo_model.baseline_interval == expected_baseline_interval
        ), "baseline_interval attribute value not correct"
