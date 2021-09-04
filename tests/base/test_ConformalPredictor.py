import numpy as np
import pandas as pd
import re

from pitci.base import ConformalPredictor
import pitci

import pytest


class DummyConformalPredictor(ConformalPredictor):
    """Dummy class inheriting from ConformalPredictor so it's
    functionality can be tested.
    """

    def __init__(self, model="abc"):

        super().__init__(model=model)

    def _generate_predictions(self, data):
        """Dummy function that returns 0s of shape (n,) where data has n rows."""

        return np.zeros(data.shape[0])

    def _calculate_scaling_factors(self, data):
        """Dummy function that returns 0s of shape (n,) where predictions has n rows."""

        return np.ones(data.shape[0])

    def _calculate_nonconformity_scores(self, predictions, response, scaling_factors):
        """Dummy function that returns 0s of shape (n,) where data has n rows."""

        return np.zeros(predictions.shape[0])


class TestInit:
    """Tests for the ConformalPredictor._init__ method."""

    def test_version_attribute_set(self):
        """Test that the version attribute is set in init."""

        dummy_confo_model = DummyConformalPredictor()

        assert (
            dummy_confo_model.__version__ == pitci.__version__
        ), "version attribute not set correctly"

    def test_model_attribute_set(self):
        """Test that the model attribute is set in init."""

        dummy_confo_model = DummyConformalPredictor(model=456)

        assert dummy_confo_model.model == 456, "model attribute not set correctly"


class TestCalibrate:
    """Tests for the ConformalPredictor.calibrate method."""

    @pytest.mark.parametrize("alpha", [(-0.0001), (-1), (1.0001), (2), (55)])
    def test_alpha_value_error(self, dmatrix_2x1_with_label, alpha):
        """Test an exception is raised if alpha is below 0 or greater than 1."""

        dummy_confo_model = DummyConformalPredictor()

        with pytest.raises(
            ValueError, match=re.escape("alpha must be in range [0 ,1]")
        ):

            dummy_confo_model.calibrate(
                data=dmatrix_2x1_with_label, alpha=alpha, response=np.array([0, 1])
            )

    def test_alpha_incorrect_type_error(self, dmatrix_2x1_with_label):
        """Test an exception is raised if alpha is not an int or float."""

        dummy_confo_model = DummyConformalPredictor()

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

        dummy_confo_model = DummyConformalPredictor()

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"response is not in expected types {[np.ndarray, pd.Series]}, got {bool}"
            ),
        ):

            dummy_confo_model.calibrate(
                data=dmatrix_2x1_with_label, alpha=0.5, response=False
            )

    def test_calibrate_interval_call(self, mocker, np_2x1_with_label):
        """Test the call to the _calibrate_interval method."""

        dummy_confo_model = DummyConformalPredictor()

        mocked = mocker.patch.object(
            pitci.base.ConformalPredictor, "_calibrate_interval"
        )

        dummy_confo_model.calibrate(
            data=np_2x1_with_label[0],
            alpha=0.1,
            response=np_2x1_with_label[1],
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

        np.testing.assert_array_equal(call_kwargs["data"], np_2x1_with_label[0])

        np.testing.assert_array_equal(call_kwargs["response"], np_2x1_with_label[1])

        assert (
            call_kwargs["alpha"] == 0.1
        ), "alpha arg incorrect in _calibrate_interval call"

    def test_alpha_attribute_set(self, np_2x1_with_label):
        """Test that the value passed in the alpha argument is set to the alpha attribute."""

        dummy_confo_model = DummyConformalPredictor()

        assert not hasattr(
            dummy_confo_model, "alpha"
        ), "confo model already has alpha attribute"

        dummy_confo_model.calibrate(
            data=np_2x1_with_label[0],
            alpha=0.125,
            response=np_2x1_with_label[1],
        )

        assert (
            dummy_confo_model.alpha == 0.125
        ), "alpha attribute not set correctly by calibrate method"


class TestPredictWithInterval:
    """Tests for the ConformalPredictor.predict_with_interval method."""

    def test_exception_no_baseline_interval(self):
        """Test an exception is raised if no baseline_interval atttibute is present."""

        dummy_confo_model = DummyConformalPredictor()

        assert not hasattr(dummy_confo_model, "baseline_interval")

        with pytest.raises(
            AttributeError,
            match="AbsoluteErrorConformalPredictor does not have baseline_interval attribute, "
            "run calibrate first.",
        ):

            dummy_confo_model.predict_with_interval(np.array([1, 0]))

    def test_expected_output(self):
        """Test the intervals returned are as expected."""

        dummy_confo_model = DummyConformalPredictor()

        dummy_confo_model.baseline_interval = 5

        results = dummy_confo_model.predict_with_interval(np.array([1, 0, -1]))

        # note scaling factors are one by and predictions zero default in the dummy class
        expected_results = np.array([[-5, 0, 5], [-5, 0, 5], [-5, 0, 5]])

        np.testing.assert_array_equal(results, expected_results)

    def test_expected_output2(self, mocker):
        """Test the intervals returned are as expected."""

        dummy_confo_model = DummyConformalPredictor()

        dummy_confo_model.baseline_interval = 3

        #  set return value from _generate_predictions
        mocker.patch.object(
            DummyConformalPredictor,
            "_generate_predictions",
            return_value=np.array([1, 2, 3, 4]),
        )

        results = dummy_confo_model.predict_with_interval(np.array([0, 0, 0, 0]))

        # note scaling factors are one by default in the dummy class
        expected_results = np.array([[-2, 1, 4], [-1, 2, 5], [0, 3, 6], [1, 4, 7]])

        np.testing.assert_array_equal(results, expected_results)

    def test_expected_output3(self, mocker):
        """Test the intervals returned are as expected."""

        dummy_confo_model = DummyConformalPredictor()

        dummy_confo_model.baseline_interval = 6

        #  set return value from _generate_predictions
        mocker.patch.object(
            DummyConformalPredictor,
            "_generate_predictions",
            return_value=np.array([1, 2, 3, 4, 5]),
        )

        #  set return value from _calculate_scaling_factors
        mocker.patch.object(
            DummyConformalPredictor,
            "_calculate_scaling_factors",
            return_value=np.array([-1, 1, 2, 0.5, 0.1]),
        )

        results = dummy_confo_model.predict_with_interval(np.array([0, 0, 0, 0, 0]))

        expected_results = np.array(
            [[7, 1, -5], [-4, 2, 8], [-9, 3, 15], [1, 4, 7], [4.4, 5, 5.6]]
        )

        np.testing.assert_array_equal(results, expected_results)


class TestLookupBaselineInterval:
    """Tests for the ConformalPredictor._lookup_baseline_interval method."""

    def test_baseline_interval_returned(self):
        """Test that the baseline interval attribute is returned from the method."""

        dummy_confo_model = DummyConformalPredictor()

        dummy_confo_model.baseline_interval = 12345

        result = dummy_confo_model._lookup_baseline_interval([])

        assert (
            result == dummy_confo_model.baseline_interval
        ), "baseline_interval not returned from _lookup_baseline_interval method"


class TestCalibrateInterval:
    """Tests for the ConformalPredictor._calibrate_interval method."""

    def test_function_calls(self, mocker):
        """Test that the chain of function calls within _calibrate_interval happen
        as expected.
        """

        dummy_confo_model = DummyConformalPredictor()

        # values to call calibrate method with
        data_value = np.array([-6, -7, -8, -9, -10])
        response_value = np.array([-1, -2, -3, -4, -5])
        alpha_value = 0.91

        # set return values from methods called within _calibrate_interval
        predictions_return_value = np.array([1, 2, 3, 4, 5])
        scaling_factors_return_value = np.array([6, 7, 8, 9, 10])
        nonconformity_return_value = np.array([11, 12, 13, 14, 15])
        nonconformity_at_alpha_return_value = 1.2345

        mocker.patch.object(
            DummyConformalPredictor,
            "_generate_predictions",
            return_value=predictions_return_value,
        )

        mocker.patch.object(
            DummyConformalPredictor,
            "_calculate_scaling_factors",
            return_value=scaling_factors_return_value,
        )

        mocker.patch.object(
            DummyConformalPredictor,
            "_calculate_nonconformity_scores",
            return_value=nonconformity_return_value,
        )

        mocker.patch.object(
            pitci.nonconformity,
            "nonconformity_at_alpha",
            return_value=nonconformity_at_alpha_return_value,
        )

        dummy_confo_model._calibrate_interval(
            data=data_value, response=response_value, alpha=alpha_value
        )

        assert (
            DummyConformalPredictor._generate_predictions.call_count == 1
        ), "_generate_predictions method not called the correct number of times"

        assert (
            DummyConformalPredictor._generate_predictions.call_args_list[0][1] == {}
        ), "keyword arguments in DummyConformalPredictor._generate_predictions call incorrect"

        assert (
            len(DummyConformalPredictor._generate_predictions.call_args_list[0][0]) == 1
        ), "number of positional arguments in DummyConformalPredictor._generate_predictions call incorrect"

        # test _generate_predictions is called with the data arg
        np.testing.assert_array_equal(
            data_value,
            DummyConformalPredictor._generate_predictions.call_args_list[0][0][0],
        )

        assert (
            DummyConformalPredictor._calculate_scaling_factors.call_count == 1
        ), "_calculate_scaling_factors method not called the correct number of times"

        assert (
            DummyConformalPredictor._calculate_scaling_factors.call_args_list[0][1]
            == {}
        ), "keyword arguments in DummyConformalPredictor._calculate_scaling_factors call incorrect"

        assert (
            len(DummyConformalPredictor._calculate_scaling_factors.call_args_list[0][0])
            == 1
        ), "number of positional arguments in DummyConformalPredictor._calculate_scaling_factors call incorrect"

        # test _calculate_scaling_factors is called with the data arg
        np.testing.assert_array_equal(
            data_value,
            DummyConformalPredictor._calculate_scaling_factors.call_args_list[0][0][0],
        )

        assert (
            DummyConformalPredictor._calculate_nonconformity_scores.call_count == 1
        ), "_calculate_nonconformity_scores method not called the correct number of times"

        assert (
            DummyConformalPredictor._calculate_nonconformity_scores.call_args_list[0][1]
            == {}
        ), "keyword arguments in DummyConformalPredictor._calculate_nonconformity_scores call incorrect"

        assert (
            len(
                DummyConformalPredictor._calculate_nonconformity_scores.call_args_list[
                    0
                ][0]
            )
            == 3
        ), "number of positional arguments in DummyConformalPredictor._calculate_nonconformity_scores call incorrect"

        # test the first arg _calculate_nonconformity_scores is called with is the output from
        # the _generate_predictions method
        np.testing.assert_array_equal(
            predictions_return_value,
            DummyConformalPredictor._calculate_nonconformity_scores.call_args_list[0][
                0
            ][0],
        )

        # test the second arg _calculate_nonconformity_scores is called with is the response arg
        np.testing.assert_array_equal(
            response_value,
            DummyConformalPredictor._calculate_nonconformity_scores.call_args_list[0][
                0
            ][1],
        )

        # test the third arg _calculate_nonconformity_scores is called with is the output from
        # the _calculate_scaling_factors method
        np.testing.assert_array_equal(
            scaling_factors_return_value,
            DummyConformalPredictor._calculate_nonconformity_scores.call_args_list[0][
                0
            ][2],
        )

        assert (
            pitci.nonconformity.nonconformity_at_alpha.call_count == 1
        ), "nonconformity.nonconformity_at_alpha method not called the correct number of times"

        assert (
            pitci.nonconformity.nonconformity_at_alpha.call_args_list[0][1] == {}
        ), "keyword arguments in nonconformity.nonconformity_at_alpha call incorrect"

        assert (
            len(pitci.nonconformity.nonconformity_at_alpha.call_args_list[0][0]) == 2
        ), "number of positional arguments in nonconformity.nonconformity_at_alpha call incorrect"

        # test the first arg nonconformity.nonconformity_at_alpha is called with is the output from
        # the _calculate_nonconformity_scores method
        np.testing.assert_array_equal(
            nonconformity_return_value,
            pitci.nonconformity.nonconformity_at_alpha.call_args_list[0][0][0],
        )

        # test the second arg nonconformity.nonconformity_at_alpha is called with is the alpha arg
        assert (
            alpha_value
            == pitci.nonconformity.nonconformity_at_alpha.call_args_list[0][0][1]
        ), "2nd positional arg in nonconformity.nonconformity_at_alpha call not correct"

        assert (
            dummy_confo_model.baseline_interval == nonconformity_at_alpha_return_value
        ), (
            "baseline_interval attribute is not set to the value returned from "
            "nonconformity.nonconformity_at_alpha function"
        )
