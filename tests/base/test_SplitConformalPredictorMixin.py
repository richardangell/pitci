import numpy as np
import abc
import re

from pitci.base import SplitConformalPredictorMixin, AbsoluteErrorConformalPredictor
import pitci

import pytest


class DummySplitConformalPredictor(
    SplitConformalPredictorMixin, AbsoluteErrorConformalPredictor
):
    """Dummy class that inherits from SplitConformalPredictorMixin how it is intended
    to be used in the package.

    This is required to initialise an instance of the class as the init method calls
    super init (object.__init__) which errors when passing the model argument to it.

    """

    def _generate_predictions(self):
        """Implement dummy method required by ConformalPredictor abstract base class."""

        pass


class TestInit:
    """Tests for the SplitConformalPredictorMixin.__init__ method."""

    def test_n_bins_not_int_error(self):
        """Test an exception is raised if n_bins is not an int."""

        with pytest.raises(
            TypeError,
            match=re.escape(f"n_bins is not in expected types {[int]}, got {str}"),
        ):

            SplitConformalPredictorMixin(n_bins="a", model="abcd")

    def test_n_bins_not_greater_than_one_error(self):
        """Test an exception is raised if n_bins is not greater than 1."""

        with pytest.raises(ValueError, match="n_bins should be greater than 1"):

            SplitConformalPredictorMixin(n_bins=1, model="abcd")

    def test_bin_attributes_set(self, mocker):
        """Test that n_bins and bin_quantiles attributes are set in init."""

        dummy_confo_model = DummySplitConformalPredictor(n_bins=15, model="abcd")

        assert dummy_confo_model.n_bins == 15, "n_bins attribute not set correctly"

        np.testing.assert_array_equal(
            dummy_confo_model.bin_quantiles, np.linspace(0, 1, 16)
        )

    def test_super_init_called(self, mocker):
        """Test that the """

        mocker.spy(AbsoluteErrorConformalPredictor, "__init__")

        model_value = "abcde"

        dummy_confo_model = DummySplitConformalPredictor(n_bins=5, model=model_value)

        expected_mro = (
            DummySplitConformalPredictor,
            SplitConformalPredictorMixin,
            AbsoluteErrorConformalPredictor,
            pitci.base.ConformalPredictor,
            abc.ABC,
            object,
        )

        assert (
            DummySplitConformalPredictor.__mro__ == expected_mro
        ), "mro not expected for DummySplitConformalPredictor"

        assert (
            AbsoluteErrorConformalPredictor.__init__.call_count == 1
        ), "super init method not called once as expected"

        # self arg
        assert AbsoluteErrorConformalPredictor.__init__.call_args_list[0][0] == (
            dummy_confo_model,
        ), "positional arguments not correct in super init call"

        assert AbsoluteErrorConformalPredictor.__init__.call_args_list[0][1] == {
            "model": model_value
        }, "keyword arguments not correct in super init call"


class TestCalibrateInterval:
    """Tests for the SplitConformalPredictorMixin._calibrate_interval method."""

    def test_exception_no_leaf_node_counts(self):
        """Test an exception is raised if no leaf_node_counts atttibute is present."""

        dummy_confo_model = DummySplitConformalPredictor(n_bins=5, model="a")

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

        dummy_confo_model = DummySplitConformalPredictor(n_bins=n_bins, model="a")
        dummy_confo_model.leaf_node_counts = {}

        data = np.array([1, 0])
        response = np.zeros(scaling_factors.shape)
        predictions = np.zeros(scaling_factors.shape)

        # set return value from _generate_predictions
        mocker.patch.object(
            DummySplitConformalPredictor,
            "_generate_predictions",
            return_value=predictions,
        )

        # set return value from _calculate_scaling_factors
        mocker.patch.object(
            DummySplitConformalPredictor,
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

        dummy_confo_model = DummySplitConformalPredictor(n_bins=5, model="a")
        dummy_confo_model.leaf_node_counts = {}

        scaling_factors = np.array([i for i in range(101)])

        data = np.array([1, 0])
        response = np.zeros(scaling_factors.shape)
        predictions = np.zeros(scaling_factors.shape)

        nonconformity_values = np.array([i for i in range(100, 201)])

        # set return value from _generate_predictions
        mocker.patch.object(
            DummySplitConformalPredictor,
            "_generate_predictions",
            return_value=predictions,
        )

        # set return value from _calculate_scaling_factors
        mocker.patch.object(
            DummySplitConformalPredictor,
            "_calculate_scaling_factors",
            return_value=scaling_factors,
        )

        # # set return value from _calculate_nonconformity_scores
        mocker.patch.object(
            DummySplitConformalPredictor,
            "_calculate_nonconformity_scores",
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
        # each bin has the 80th percentile (with interpolation up to the
        # higher value if required) calculated which give the values below;
        expected_baseline_intervals = np.array([116, 137, 157, 177, 197])

        np.testing.assert_array_equal(
            dummy_confo_model.baseline_interval, expected_baseline_intervals
        )


class TestLookupBaselineInterval:
    """Tests for the SplitConformalPredictorMixin._lookup_baseline_interval method."""

    def test_correct_interval_lookuped_up(self):
        """Test that the correct value from baseline_intervals is returned."""

        dummy_confo_model = DummySplitConformalPredictor(n_bins=4, model="a")

        dummy_confo_model.baseline_interval = np.array([2, 4, 6, 8])
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

        dummy_confo_model = DummySplitConformalPredictor(n_bins=4, model="a")

        dummy_confo_model.baseline_interval = baseline_intervals

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

        dummy_confo_model = DummySplitConformalPredictor(n_bins=6, model="a")

        dummy_confo_model.baseline_interval = baseline_intervals

        with pytest.warns(None) as warnings:

            dummy_confo_model._check_interval_monotonicity()

        assert (
            len(warnings) == 0
        ), "warnings were raised when baseline_intervals were monotonic"
