import numpy as np
import re

from pitci.base import SplitConformalPredictorMixin
import pitci

import pytest


class DummySplitConformalPredictorMixin(SplitConformalPredictorMixin):
    """Dummy class inheriting from SplitConformalPredictorMixin so it's
    functionality can be tested.
    """

    def __init__(self, model="abcd", n_bins=3):
        """Dummy init method that only calls SplitConformalPredictorMixin
        init method.
        """

        super().__init__(model=model, n_bins=n_bins)

    def _generate_predictions(self, data):
        """Dummy function that returns 0s of shape (n,) where data has n rows."""

        return np.zeros(data.shape[0])

    def _generate_leaf_node_predictions(self, data):
        """Dummy function for returning leaf node index predictions, not implemented in
        DummySplitConformalPredictorMixin so it has to be implemented specifically in
        each test requiring it.
        """

        raise NotImplementedError(
            "_generate_leaf_node_predictions not implemented in SplitConformalPredictorMixin"
        )


class TestInit:
    """Tests for the SplitConformalPredictorMixin.__init__ method."""

    def test_n_bins_not_int_error(self):
        """Test an exception is raised if n_bins is not an int."""

        with pytest.raises(
            TypeError,
            match=re.escape(f"n_bins is not in expected types {[int]}, got {str}"),
        ):

            DummySplitConformalPredictorMixin(n_bins="a")

    def test_n_bins_not_greater_than_one_error(self):
        """Test an exception is raised if n_bins is not greater than 1."""

        with pytest.raises(ValueError, match="n_bins should be greater than 1"):

            DummySplitConformalPredictorMixin(n_bins=1)

    def test_bin_attributes_set(self):
        """Test that n_bins and bin_quantiles attributes are set in init."""

        dummy_confo_model = DummySplitConformalPredictorMixin(n_bins=15)

        assert dummy_confo_model.n_bins == 15, "n_bins attribute not set correctly"

        np.testing.assert_array_equal(
            dummy_confo_model.bin_quantiles, np.linspace(0, 1, 16)
        )


class TestCalibrateInterval:
    """Tests for the SplitConformalPredictorMixin._calibrate_interval method."""

    def test_exception_no_leaf_node_counts(self):
        """Test an exception is raised if no leaf_node_counts atttibute is present."""

        dummy_confo_model = DummySplitConformalPredictorMixin(n_bins=5)

        assert not hasattr(dummy_confo_model, "leaf_node_counts")

        with pytest.raises(
            AttributeError,
            match="object does not have leaf_node_counts attribute, run calibrate first.",
        ):

            dummy_confo_model._calibrate_interval(np.array([1, 0]), np.array([1, 0]))

    @pytest.mark.parametrize(
        "n_bins, scaling_factors, expected_scaling_factor_cut_points",
        [
            (2, np.array([1, 0, 2, 4, 5, 3]), np.array([0, 2.5, 5])),
            (3, np.array([6, 1, 0, 2, 4, 5, 3]), np.array([0, 2, 4, 6])),
            (
                4,
                np.array([6, 1, 0, 2, 4, 5, 3, 7, 8, 9, 10]),
                np.array([0, 2.5, 5, 7.5, 10]),
            ),
        ],
    )
    def test_scaling_factor_cut_points_expected(
        self, mocker, n_bins, scaling_factors, expected_scaling_factor_cut_points
    ):
        """Test that scaling_factor_cut_points attribute is calculated as expected."""

        dummy_confo_model = DummySplitConformalPredictorMixin(n_bins=n_bins)
        dummy_confo_model.leaf_node_counts = {}

        data = np.array([1, 0])
        response = np.zeros(scaling_factors.shape)
        predictions = np.zeros(scaling_factors.shape)

        # set return value from _generate_predictions
        mocker.patch.object(
            DummySplitConformalPredictorMixin,
            "_generate_predictions",
            return_value=predictions,
        )

        # set return value from _calculate_scaling_factors
        mocker.patch.object(
            DummySplitConformalPredictorMixin,
            "_calculate_scaling_factors",
            return_value=scaling_factors,
        )

        dummy_confo_model._calibrate_interval(data=data, response=response)

        np.testing.assert_array_equal(
            dummy_confo_model.scaling_factor_cut_points,
            expected_scaling_factor_cut_points,
        )

    def test_baseline_intervals_expected(self, mocker):
        """Test that baseline_intervals attribute is calculated as expected."""

        dummy_confo_model = DummySplitConformalPredictorMixin(n_bins=5)
        dummy_confo_model.leaf_node_counts = {}

        scaling_factors = np.array([i for i in range(101)])

        data = np.array([1, 0])
        response = np.zeros(scaling_factors.shape)
        predictions = np.zeros(scaling_factors.shape)

        nonconformity_values = np.array([i for i in range(100, 201)])

        # set return value from _generate_predictions
        mocker.patch.object(
            DummySplitConformalPredictorMixin,
            "_generate_predictions",
            return_value=predictions,
        )

        # set return value from _calculate_scaling_factors
        mocker.patch.object(
            DummySplitConformalPredictorMixin,
            "_calculate_scaling_factors",
            return_value=scaling_factors,
        )

        # set return value from _generate_predictions
        mocker.patch.object(
            pitci.nonconformity,
            "scaled_absolute_error",
            return_value=nonconformity_values,
        )

        dummy_confo_model._calibrate_interval(data=data, response=response, alpha=0.8)

        expected_scaling_factor_cut_points = np.array(
            [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
        )

        np.testing.assert_array_almost_equal(
            dummy_confo_model.scaling_factor_cut_points,
            expected_scaling_factor_cut_points,
        )

        # in terms of bins that the nonconformity_values fall into
        # the first bin will contain values [100:120], the second [121:140]
        # the third [141:160] and so on up to 5 bins
        # each bin has the 80th percentile calculated which give
        # the values below
        expected_baseline_intervals = np.array([116, 136, 156, 176, 196])

        np.testing.assert_array_equal(
            dummy_confo_model.baseline_intervals, expected_baseline_intervals
        )


class TestPredictWithInterval:
    """Tests for the LeafNodeScaledConformalPredictor.predict_with_interval method."""

    def test_exception_no_baseline_interval(self):
        """Test an exception is raised if no baseline_intervals atttibute is present."""

        dummy_confo_model = DummySplitConformalPredictorMixin(n_bins=5)

        assert not hasattr(dummy_confo_model, "baseline_intervals")

        with pytest.raises(
            AttributeError,
            match="object does not have baseline_intervals attribute, run calibrate first.",
        ):

            dummy_confo_model.predict_with_interval(np.array([1, 0]))

    def test_expected_output(self, mocker):
        """Test the intervals returned are calculated as predictions +-
        (scaling factor * baseline interval).
        """

        dummy_confo_model = DummySplitConformalPredictorMixin()

        dummy_confo_model.baseline_intervals = 2

        # set return value from _generate_predictions
        mocker.patch.object(
            DummySplitConformalPredictorMixin,
            "_generate_predictions",
            return_value=np.array([-4, 0, 1, 4]),
        )

        # set return value from _generate_predictions
        mocker.patch.object(
            DummySplitConformalPredictorMixin,
            "_calculate_scaling_factors",
            return_value=np.array([0.5, 1, 2, -2]),
        )

        # set return value from _lookup_baseline_interval
        mocker.patch.object(
            DummySplitConformalPredictorMixin,
            "_lookup_baseline_interval",
            return_value=np.array([0, 0.5, 4, 10]),
        )

        results = dummy_confo_model.predict_with_interval(np.array([1, 0, -1]))

        expected_results = np.array(
            [[-4, -4, -4], [-0.5, 0, 0.5], [-7, 1, 9], [24, 4, -16]]
        )

        np.testing.assert_array_equal(results, expected_results)


class TestLookupBaselineInterval:
    """Tests for the SplitConformalPredictorMixin._lookup_baseline_interval method."""

    def test_correct_interval_lookuped_up(self):
        """Test that the correct value from baseline_intervals is returned."""

        dummy_confo_model = DummySplitConformalPredictorMixin(n_bins=4)

        dummy_confo_model.baseline_intervals = np.array([2, 4, 6, 8])
        dummy_confo_model.scaling_factor_cut_points = np.array([-10, 0, 10, 20, 30])

        s = 0.00000001

        lookup_scaling_factor_values = np.array(
            [
                -11,
                -10 - s,
                -10,
                -10 + s,
                -5,
                0 - s,
                0,
                0 + s,
                5,
                10 - s,
                10,
                10 + s,
                15,
                20 - s,
                20,
                20 + s,
                25,
                30 - s,
                30,
                30 + s,
                35,
            ]
        )

        expected_looked_up_intervals = np.array(
            [2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8]
        )

        looked_up_intervals = dummy_confo_model._lookup_baseline_interval(
            lookup_scaling_factor_values
        )

        np.testing.assert_array_equal(expected_looked_up_intervals, looked_up_intervals)


class TestCheckIntervalMonotonicity:
    """Tests for the SplitConformalPredictorMixin._check_interval_monotonicity method."""

    @pytest.mark.parametrize(
        "baseline_intervals",
        [(np.array([1, 2, 3, 4, 3])), (np.array([1, 0.2, 3, 4, 300]))],
    )
    def test_warning_raised(self, baseline_intervals):
        """Test the correct warning is raised if baseline_intervals are not monotonic."""

        dummy_confo_model = DummySplitConformalPredictorMixin(n_bins=4)

        dummy_confo_model.baseline_intervals = baseline_intervals

        with pytest.warns(
            Warning,
            match="baseline intervals calculated on 4 bins are not monotonic in either direction",
        ):

            dummy_confo_model._check_interval_monotonicity()

    @pytest.mark.parametrize(
        "baseline_intervals",
        [(np.array([0.1, 0.2, 0.3, 0.4, 20])), (np.array([-1, -2, -3, -4, -300]))],
    )
    def test_no_warnings_when_monotonic(self, baseline_intervals):
        """Test no warnings are raised when baseline_intervals are monotonic."""

        dummy_confo_model = DummySplitConformalPredictorMixin(n_bins=6)

        dummy_confo_model.baseline_intervals = baseline_intervals

        with pytest.warns(None) as warnings:

            dummy_confo_model._check_interval_monotonicity()

        assert (
            len(warnings) == 0
        ), "warnings were raised when baseline_intervals were monotonic"
