import numpy as np
import xgboost as xgb

from pitci.base import AbsoluteErrorConformalPredictor
import pitci

import pytest


class DummyAbsoluteErrorConformalPredictor(AbsoluteErrorConformalPredictor):
    """Dummy class inheriting from AbsoluteErrorConformalPredictor so it's
    functionality can be tested.
    """

    def __init__(self, model="abc"):

        super().__init__(model=model)

    def _generate_predictions(self, data):
        """Dummy function that returns 0s of shape (n,) where data has n rows."""

        return np.zeros(data.num_row())


def test_inheritance():
    """Test AbsoluteErrorConformalPredictor inherits from ConformalPredictor."""

    dummy_confo_model = DummyAbsoluteErrorConformalPredictor()

    assert isinstance(
        dummy_confo_model, pitci.base.ConformalPredictor
    ), "AbsoluteErrorConformalPredictor not inheriting from ConformalPredictor"


class TestCalculateNonconformityScores:
    """Tests for the AbsoluteErrorConformalPredictor._calculate_nonconformity_scores method."""

    def test_scaled_absolute_error_call(self, mocker):
        """Test the nonconformity.scaled_absolute_error function is called correctly."""

        dummy_confo_model = DummyAbsoluteErrorConformalPredictor()

        nonconformity_scores_return_value = 1234
        predictions_value = 1
        response_value = 2
        scaling_factors_value = 3

        mocker.patch.object(
            pitci.nonconformity,
            "scaled_absolute_error",
            return_value=nonconformity_scores_return_value,
        )

        result = dummy_confo_model._calculate_nonconformity_scores(
            predictions_value, response_value, scaling_factors_value
        )

        assert (
            pitci.nonconformity.scaled_absolute_error.call_count == 1
        ), "nonconformity.scaled_absolute_error function not called the correct number of times"

        assert (
            pitci.nonconformity.scaled_absolute_error.call_args_list[0][0] == ()
        ), "positional arguments in nonconformity.scaled_absolute_error call incorrect"

        expected_call_kwargs = {
            "predictions": predictions_value,
            "response": response_value,
            "scaling": scaling_factors_value,
        }

        assert (
            pitci.nonconformity.scaled_absolute_error.call_args_list[0][1]
            == expected_call_kwargs
        ), "keyword arguments in nonconformity.scaled_absolute_error call incorrect"

        assert result == nonconformity_scores_return_value, (
            "return value from _calculate_nonconformity_scores is not the output from "
            "nonconformity.scaled_absolute_error function"
        )


class TestCalculateScalingFactors:
    """Tests for the AbsoluteErrorConformalPredictor._calculate_scaling_factors method."""

    @pytest.mark.parametrize(
        "data",
        [
            (np.array([3])),
            (np.array([3, 9])),
            (np.ones((5, 6))),
        ],
    )
    def test_return_value(self, data):
        """Test 1 is always returned from the method."""

        dummy_confo_model = DummyAbsoluteErrorConformalPredictor()

        result = dummy_confo_model._calculate_scaling_factors(data)

        assert result == 1, "output from _calculate_scaling_factors not correct"


class TestCalibrateInterval:
    """Tests for the AbsoluteErrorConformalPredictor._calibrate_interval method.

    Note, AbsoluteErrorConformalPredictor implements the _calculate_nonconformity_scores
    method so these tests are tests that the baseline_interval is calculated
    correctly given some inputs.

    """

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
