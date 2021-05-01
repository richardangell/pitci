import numpy as np
import re
import pitci.helpers as helpers

import pytest


class TestGatherIntervals:
    """Tests for the gather_intervals function."""

    @pytest.mark.parametrize(
        "lower_interval, upper_interval, intervals_with_predictions",
        [(None, None, None), (None, 1, None), (1, None, None)],
    )
    def test_both_none(
        self, lower_interval, upper_interval, intervals_with_predictions
    ):
        """Test an exception is raised if both the inidividual intervals and combined arguments are None."""

        with pytest.raises(
            ValueError,
            match=re.escape(
                "either lower_interval and upper_interval or intervals_with_predictions must"
                "be specified but both are None"
            ),
        ):

            helpers.gather_intervals(
                lower_interval, upper_interval, intervals_with_predictions
            )

    @pytest.mark.parametrize(
        "lower_interval, upper_interval, intervals_with_predictions",
        [(1, 1, 1), (1, None, 1), (None, 1, 1)],
    )
    def test_both_not_none(
        self, lower_interval, upper_interval, intervals_with_predictions
    ):
        """Test an exception is raised if both the inidividual intervals and combined arguments are not None."""

        with pytest.raises(
            ValueError,
            match=re.escape(
                "either lower_interval and upper_interval or intervals_with_predictions must"
                "be specified but both are specified"
            ),
        ):

            helpers.gather_intervals(
                lower_interval, upper_interval, intervals_with_predictions
            )

    def test_intervals_with_predictions_not_3_columns_error(self):
        """Test an exeption is raised if intervals_with_predictions does not have 3 columns."""

        with pytest.raises(
            ValueError,
            match=re.escape("expecting intervals_with_predictions to have 3 columns"),
        ):

            helpers.gather_intervals(
                intervals_with_predictions=np.array([[1, 2], [1, 2]])
            )

    def test_lower_upper_intervals_different_rows_error(self):
        """Test and exception is raised if lower and upper intervals do not have the same number of rows."""

        with pytest.raises(
            ValueError,
            match=re.escape(
                "lower_interval_return and upper_interval_return have different shapes"
            ),
        ):

            helpers.gather_intervals(
                lower_interval=np.array([1, 2, 1, 2]),
                upper_interval=np.array([1, 2, 3]),
            )

    def test_output_intervals_with_predictions_input(self):
        """Test the output if intervals_with_predictions passed."""

        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        lower_out, upper_out = helpers.gather_intervals(intervals_with_predictions=arr)

        np.testing.assert_array_equal(lower_out, arr[:, 0])

        np.testing.assert_array_equal(upper_out, arr[:, 2])

    def test_output_individual_intervals_input(self):
        """Test the inputs are returned if lower_interval and upper_interval are passed."""

        arr_lower = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        arr_upper = np.array([5, 6, 7, 8, 9, 4, 3, 2, 1])

        lower_out, upper_out = helpers.gather_intervals(
            lower_interval=arr_lower, upper_interval=arr_upper
        )

        np.testing.assert_array_equal(lower_out, arr_lower)

        np.testing.assert_array_equal(upper_out, arr_upper)


class TestCheckResponseWithinInterval:
    """Tests for the check_response_within_interval function."""

    @pytest.mark.parametrize(
        "lower_interval, upper_interval, intervals_with_predictions",
        [
            (None, None, np.array([[1, 2, 3]])),
            (np.array([1, 2, 3]), np.array([1, 2, 3]), None),
        ],
    )
    def test_different_rows_error(
        self, lower_interval, upper_interval, intervals_with_predictions
    ):
        """Test an exception is raised if response and intervals have different numbers of rows."""

        response = np.array([1, 2])

        with pytest.raises(
            ValueError,
            match=re.escape("response and intervals have different numbers of rows"),
        ):

            helpers.check_response_within_interval(
                response, lower_interval, upper_interval, intervals_with_predictions
            )

    @pytest.mark.parametrize(
        "response, lower_interval, upper_interval, expected_within, expected_outside",
        [
            (
                np.array([1, 2, 3]),
                np.array([-1, -2, -1]),
                np.array([4, 5, 10]),
                1,
                None,
            ),
            (np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([4, 5, 10]), 1, None),
            (np.array([1, 2, 3]), np.array([-1, -2, -1]), np.array([1, 2, 3]), 1, None),
            (
                np.array([-1, -2, -1]),
                np.array([1, 2, 3]),
                np.array([4, 5, 10]),
                None,
                1,
            ),
            (
                np.array([4, 5, 10]),
                np.array([-1, -2, -1]),
                np.array([1, 2, 3]),
                None,
                1,
            ),
            (
                np.array([1, 2, 50]),
                np.array([-1, -2, -1]),
                np.array([4, 5, 10]),
                2 / 3,
                1 / 3,
            ),
            (
                np.array([1, 500, 50]),
                np.array([-1, -2, -1]),
                np.array([4, 5, 10]),
                1 / 3,
                2 / 3,
            ),
        ],
    )
    def test_output(
        self,
        response,
        lower_interval,
        upper_interval,
        expected_within,
        expected_outside,
    ):
        """Test the proportions output from the function are expected."""

        results = helpers.check_response_within_interval(
            response=response,
            lower_interval=lower_interval,
            upper_interval=upper_interval,
        )

        if expected_within is not None:

            assert (
                results[True] == expected_within
            ), "proportion values for response within interval not correct when passing individual intervals"

        if expected_outside is not None:

            assert (
                results[False] == expected_outside
            ), "proportion values for response outside interval not correct when passing individual intervals"

        combined_intervals = (
            np.concatenate(
                [lower_interval, np.zeros(lower_interval.shape), upper_interval], axis=0
            )
            .reshape((lower_interval.shape[0], 3))
            .T
        )

        results = helpers.check_response_within_interval(
            response=response, intervals_with_predictions=combined_intervals
        )

        if expected_within is not None:

            assert (
                results[True] == expected_within
            ), "proportion values for response within interval not correct when passing individual intervals"

        if expected_outside is not None:

            assert (
                results[False] == expected_outside
            ), "proportion values for response outside interval not correct when passing individual intervals"


class TestCheckIntervalWidth:
    """Tests for the check_interval_width function."""

    def test_output(self):
        """Test that check_interval_width is giving the expected outputs."""

        upper_interval = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        lower_interval = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        quantiles = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        results = helpers.check_interval_width(
            lower_interval, upper_interval, None, quantiles
        )

        for v in upper_interval.tolist():

            assert results[v / 10] == v, f"incorrect quantile {v/10}"

        assert results["iqr"] == 5, "incorrect iqr for interval distribution"
        assert results["mean"] == 5, "incorrect mean for interval distribution"
        assert results["std"] == np.std(
            upper_interval, ddof=1
        ), "incorrect std for interval distribution"

        combined_intervals = np.concatenate(
            [
                lower_interval.reshape((lower_interval.shape[0], 1)),
                np.zeros(lower_interval.shape).reshape((lower_interval.shape[0], 1)),
                upper_interval.reshape((lower_interval.shape[0], 1)),
            ],
            axis=1,
        )

        results = helpers.check_interval_width(
            None, None, combined_intervals, quantiles
        )

        for v in upper_interval.tolist():

            assert results[v / 10] == v, f"incorrect quantile {v/10}"

        assert results["iqr"] == 5, "incorrect iqr for interval distribution"
        assert results["mean"] == 5, "incorrect mean for interval distribution"
        assert results["std"] == np.std(
            upper_interval, ddof=1
        ), "incorrect std for interval distribution"


class TestPreparePredictionIntervalDf:
    """Tests for the prepare_prediction_interval_df function."""

    def test_array_added_to_df(self):
        """Test that array values are added to the output in the lower, prediction
        and upper columns.
        """

        arr = np.array([[1, 2, 3], [4, 5, 6]])

        response = np.array([9, 10])

        results = helpers.prepare_prediction_interval_df(
            intervals_with_predictions=arr, response=response
        )

        np.testing.assert_array_equal(
            results[["lower", "prediction", "upper"]].values, arr
        )

    def test_response_added_to_df(self):
        """Test that the values passed in response are added to the output in the
        response column.
        """

        arr = np.array([[1, 2, 3], [4, 5, 6]])

        response = np.array([[9], [10]])

        results = helpers.prepare_prediction_interval_df(
            intervals_with_predictions=arr, response=response
        )

        np.testing.assert_array_equal(results[["response"]].values, response)
