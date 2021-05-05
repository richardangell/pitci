import numpy as np
import pandas as pd
import re

from pitci.base import LeafNodeScaledConformalPredictor
import pitci

import pytest
from unittest.mock import Mock


class DummyLeafNodeScaledConformalPredictor(LeafNodeScaledConformalPredictor):
    """Dummy class inheriting from LeafNodeScaledConformalPredictor so it's
    functionality can be tested.
    """

    def __init__(self):
        """Dummy init method that only calls LeafNodeScaledConformalPredictor
        init method.
        """

        super().__init__()

    def _generate_predictions(self, data):
        """Dummy function that returns 0s of shape (n,) where data has n rows."""

        return np.zeros(data.shape[0])

    def _generate_leaf_node_predictions(self, data):
        """Dummy function for returning leaf node index predictions, not implemented in
        DummyLeafNodeScaledConformalPredictor so it has to be implemented specifically in
        each test requiring it.
        """

        raise NotImplementedError(
            "_generate_leaf_node_predictions not implemented in DummyLeafNodeScaledConformalPredictor"
        )


class TestInit:
    """Tests for the LeafNodeScaledConformalPredictor._init__ method."""

    def test_version_attribute_set(self):
        """Test that the version attribute is set in init."""

        dummy_confo_model = DummyLeafNodeScaledConformalPredictor()

        assert (
            dummy_confo_model.__version__ == pitci.__version__
        ), "version attribute not set correctly"


class TestCalibrate:
    """Tests for the LeafNodeScaledConformalPredictor.calibrate method."""

    @pytest.mark.parametrize("alpha", [(-0.0001), (-1), (1.0001), (2), (55)])
    def test_alpha_value_error(self, dmatrix_2x1_with_label, alpha):
        """Test an exception is raised if alpha is below 0 or greater than 1."""

        dummy_confo_model = DummyLeafNodeScaledConformalPredictor()

        with pytest.raises(
            ValueError, match=re.escape("alpha must be in range [0 ,1]")
        ):

            dummy_confo_model.calibrate(
                data=dmatrix_2x1_with_label, alpha=alpha, response=np.array([0, 1])
            )

    def test_alpha_incorrect_type_error(self, dmatrix_2x1_with_label):
        """Test an exception is raised if alpha is not an int or float."""

        dummy_confo_model = DummyLeafNodeScaledConformalPredictor()

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

        dummy_confo_model = DummyLeafNodeScaledConformalPredictor()

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"response is not in expected types {[pd.Series, np.ndarray]}, got {bool}"
            ),
        ):

            dummy_confo_model.calibrate(
                data=dmatrix_2x1_with_label, alpha=0.5, response=False
            )

    def test_calibrate_calls(self, mocker, dmatrix_2x1_with_label):
        """Test the calls to _calibrate_interval and _calibrate_leaf_node_counts methods."""

        dummy_confo_model = DummyLeafNodeScaledConformalPredictor()

        mock_manager = Mock()

        mocked = mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor, "_calibrate_leaf_node_counts"
        )

        mocked2 = mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor, "_calibrate_interval"
        )

        mock_manager.attach_mock(mocked, "the_calibrate_leaf_node_counts")
        mock_manager.attach_mock(mocked2, "the_calibrate_interval")

        dummy_confo_model.calibrate(
            data=dmatrix_2x1_with_label,
            alpha=0.1,
            response=dmatrix_2x1_with_label.get_label(),
        )

        # test each function is called the correct number of times

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to _calibrate_leaf_node_counts"

        assert (
            mocked2.call_count == 1
        ), "incorrect number of calls to _calibrate_interval"

        # test the order of calls to functions

        assert (
            mock_manager.mock_calls[0][0] == "the_calibrate_leaf_node_counts"
        ), "_calibrate_leaf_node_counts not called first"

        assert (
            mock_manager.mock_calls[1][0] == "the_calibrate_interval"
        ), "_calibrate_interval not called second"

        # test the argumnets in the _calibrate_leaf_node_counts call

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            call_pos_args == ()
        ), "positional args incorrect in _calibrate_leaf_node_counts call"

        assert (
            call_kwargs["data"] == dmatrix_2x1_with_label
        ), "data arg incorrect in _calibrate_leaf_node_counts call"

        # test the arguments in the _calibrate_interval call

        call_args = mocked2.call_args_list[0]
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
    """Tests for the LeafNodeScaledConformalPredictor.predict_with_interval method."""

    def test_exception_no_baseline_interval(self):
        """Test an exception is raised if no baseline_interval atttibute is present."""

        dummy_confo_model = DummyLeafNodeScaledConformalPredictor()

        assert not hasattr(dummy_confo_model, "baseline_interval")

        with pytest.raises(
            AttributeError,
            match="LeafNodeScaledConformalPredictor does not have baseline_interval attribute, "
            "run calibrate first.",
        ):

            dummy_confo_model.predict_with_interval(np.array([1, 0]))

    def test_expected_output(self, mocker):
        """Test the intervals returned are calculated as predictions +-
        (scaling factor * baseline interval).
        """

        dummy_confo_model = DummyLeafNodeScaledConformalPredictor()

        dummy_confo_model.baseline_interval = 2

        # set return value from _generate_predictions
        mocker.patch.object(
            DummyLeafNodeScaledConformalPredictor,
            "_generate_predictions",
            return_value=np.array([-4, 0, 1, 4]),
        )

        # set return value from _generate_predictions
        mocker.patch.object(
            DummyLeafNodeScaledConformalPredictor,
            "_calculate_scaling_factors",
            return_value=np.array([0.5, 1, 2, -2]),
        )

        results = dummy_confo_model.predict_with_interval(np.array([1, 0, -1]))

        expected_results = np.array([[-5, -4, -3], [-2, 0, 2], [-3, 1, 5], [8, 4, 0]])

        np.testing.assert_array_equal(results, expected_results)


class TestCalculateScalingFactors:
    """Tests for the LeafNodeScaledConformalPredictor._calculate_scaling_factors method."""

    def test_generate_leaf_node_predictions(self, mocker):
        """Test _generate_leaf_node_predictions is called with the data arg and the output
        from this method is passed to the _count_leaf_node_visits_from_calibration
        method.
        """

        leaf_nodes_return_value = np.array([1, 0, 1 / 3, 2])

        # set return value from _generate_leaf_node_predictions
        mocked = mocker.patch.object(
            DummyLeafNodeScaledConformalPredictor,
            "_generate_leaf_node_predictions",
            return_value=leaf_nodes_return_value,
        )

        mocked2 = mocker.patch.object(
            DummyLeafNodeScaledConformalPredictor,
            "_count_leaf_node_visits_from_calibration",
            return_value=np.array([1]),
        )

        dummy_confo_model = DummyLeafNodeScaledConformalPredictor()

        data_arg = np.array([0, 1, 3, -9])

        dummy_confo_model._calculate_scaling_factors(data_arg)

        # test the call to _generate_leaf_node_predictions

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to _generate_leaf_node_predictions"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            call_kwargs == {}
        ), "keyword args incorrect in _generate_leaf_node_predictions call"

        assert len(call_pos_args) == 1, "incorrect number of positional args"

        np.testing.assert_array_equal(call_pos_args[0], data_arg)

        # test _count_leaf_node_visits_from_calibration called with
        # _generate_leaf_node_predictions outputs

        assert (
            mocked2.call_count == 1
        ), "incorrect number of calls to _count_leaf_node_visits_from_calibration"

        call_args = mocked2.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            call_pos_args == ()
        ), "positional args incorrect in _count_leaf_node_visits_from_calibration call"

        assert list(call_kwargs.keys()) == [
            "leaf_node_predictions"
        ], "incorrect kwargs in _count_leaf_node_visits_from_calibration call"

        np.testing.assert_array_equal(
            call_kwargs["leaf_node_predictions"], leaf_nodes_return_value
        )

    def test_expected_output(self, mocker):
        """Test that the output from the function is calculated as 1 / _count_leaf_node_visits_from_calibration
        method output.
        """

        count_leaf_nodes_return_value = np.array([-4, 0, 1 / 3, 2])

        # set return value from _count_leaf_node_visits_from_calibration
        mocker.patch.object(
            DummyLeafNodeScaledConformalPredictor,
            "_count_leaf_node_visits_from_calibration",
            return_value=count_leaf_nodes_return_value,
        )

        # mock _generate_leaf_node_predictions so it doesn't run
        mocker.patch.object(
            DummyLeafNodeScaledConformalPredictor, "_generate_leaf_node_predictions"
        )

        expected_results = 1 / count_leaf_nodes_return_value

        dummy_confo_model = DummyLeafNodeScaledConformalPredictor()

        results = dummy_confo_model._calculate_scaling_factors(np.array([0]))

        np.testing.assert_array_equal(results, expected_results)
