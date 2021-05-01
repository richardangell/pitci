import numpy as np
import xgboost as xgb
import re

from pitci import AbsoluteErrorConformalPredictor
import pitci

import pytest


class TestInit:
    """Tests for the AbsoluteErrorConformalPredictor._init__ method."""

    def test_attributes_set(self, xgboost_1_split_1_tree):
        """Test that version and booster attributes are set."""

        confo_model = AbsoluteErrorConformalPredictor(xgboost_1_split_1_tree)

        assert (
            confo_model.__version__ == pitci.__version__
        ), "__version__ attribute not set to package version value"

        assert (
            confo_model.booster == xgboost_1_split_1_tree
        ), "booster attribute not set with the value passed in init"

        assert hasattr(
            confo_model, "SUPPORTED_OBJECTIVES"
        ), "AbsoluteErrorConformalPredictor does not have SUPPORTED_OBJECTIVES attribute"

    @pytest.mark.parametrize(
        "non_supported_objective",
        [
            ("survival:cox"),
            ("multi:softmax"),
            ("multi:softprob"),
            ("rank:pairwise"),
            ("rank:ndcg"),
            ("rank:map"),
        ],
    )
    def test_non_supported_objectives(
        self, dmatrix_2x1_with_label, non_supported_objective
    ):
        """Test an exception is raised if a model with a non-supported objective is passed."""

        params = {"objective": non_supported_objective}

        if "multi" in non_supported_objective:

            params["num_class"] = 3

        model = xgb.train(
            params=params, dtrain=dmatrix_2x1_with_label, num_boost_round=1
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"booster objective not supported\n{non_supported_objective} not in allowed values; {AbsoluteErrorConformalPredictor.SUPPORTED_OBJECTIVES}"
            ),
        ):

            AbsoluteErrorConformalPredictor(model)

    @pytest.mark.parametrize(
        "supported_objective",
        [
            (objective)
            for objective in AbsoluteErrorConformalPredictor.SUPPORTED_OBJECTIVES
        ],
    )
    def test_supported_objectives(
        self, dmatrix_2x1_with_label, dmatrix_2x1_with_label_gamma, supported_objective
    ):
        """Test a AbsoluteErrorConformalPredictor object can be initialised with supported objectives."""

        if "gamma" in supported_objective:

            model = xgb.train(
                params={"objective": supported_objective},
                dtrain=dmatrix_2x1_with_label_gamma,
                num_boost_round=1,
            )

        else:

            model = xgb.train(
                params={"objective": supported_objective},
                dtrain=dmatrix_2x1_with_label,
                num_boost_round=1,
            )

        try:

            AbsoluteErrorConformalPredictor(model)

        except Exception:

            pytest.fail(
                f"unable to initialise AbsoluteErrorConformalPredictor with xgb.Booster with {supported_objective} objective"
            )


class TestCalibrate:
    """Tests for the AbsoluteErrorConformalPredictor.calibrate method."""

    @pytest.mark.parametrize("alpha", [(-0.0001), (-1), (1.0001), (2), (55)])
    def test_alpha_value_error(
        self, dmatrix_2x1_with_label, xgboost_1_split_1_tree, alpha
    ):
        """Test an exception is raised if alpha is below 0 or greater than 1."""

        confo_model = AbsoluteErrorConformalPredictor(xgboost_1_split_1_tree)

        with pytest.raises(
            ValueError, match=re.escape("alpha must be in range [0 ,1]")
        ):

            confo_model.calibrate(data=dmatrix_2x1_with_label, alpha=alpha)

    @pytest.mark.parametrize(
        "attribute_name",
        [("alpha"), ("baseline_interval")],
    )
    def test_attributes_set(
        self, dmatrix_2x1_with_label, xgboost_1_split_1_tree, attribute_name
    ):
        """Test attributes are set after running method."""

        confo_model = AbsoluteErrorConformalPredictor(xgboost_1_split_1_tree)

        assert not hasattr(
            confo_model, attribute_name
        ), f"AbsoluteErrorConformalPredictor already has {attribute_name} before running set_attributes"

        confo_model.calibrate(dmatrix_2x1_with_label)

        assert hasattr(
            confo_model, attribute_name
        ), f"AbsoluteErrorConformalPredictor does not have {attribute_name} after running set_attributes"


class TestPredictWithInterval:
    """Tests for the AbsoluteErrorConformalPredictor.predict_with_interval method."""

    @pytest.mark.parametrize("n_input_rows", [(1), (2), (5), (100)])
    def test_output_shape_single_column_input(
        self, dmatrix_2x1_with_label, xgboost_1_split_1_tree, n_input_rows
    ):
        """Test the output from the function has the same number of rows as the input and 3 columns."""

        confo_model = AbsoluteErrorConformalPredictor(xgboost_1_split_1_tree)

        confo_model.calibrate(dmatrix_2x1_with_label)

        data_point = [0]

        data_column_list = []

        for i in range(n_input_rows):
            data_column_list.append(data_point)

        xgb_data_column = xgb.DMatrix(data=np.array(data_column_list))

        predictions = confo_model.predict_with_interval(xgb_data_column)

        assert predictions.shape == (
            n_input_rows,
            3,
        ), "incorrect shape for output of predict_with_interval"

    @pytest.mark.parametrize("n_input_rows", [(1), (2), (8), (58)])
    def test_output_shape_two_column_input(
        self, dmatrix_4x2_with_label, xgboost_2_split_1_tree, n_input_rows
    ):
        """Test the output from the function has the same number of rows as the input and 3 columns."""

        confo_model = AbsoluteErrorConformalPredictor(xgboost_2_split_1_tree)

        confo_model.calibrate(dmatrix_4x2_with_label)

        data_point = [0, 0]

        data_column_list = []

        for i in range(n_input_rows):
            data_column_list.append(data_point)

        xgb_data_column = xgb.DMatrix(data=np.array(data_column_list))

        predictions = confo_model.predict_with_interval(xgb_data_column)

        assert predictions.shape == (
            n_input_rows,
            3,
        ), "incorrect shape for output of predict_with_interval"

    def test_middle_output_columns_expected(
        self, mocker, dmatrix_4x2_with_label, xgboost_2_split_1_tree
    ):
        """Test that the 2nd column in the output is the output from booster.predict."""

        confo_model = AbsoluteErrorConformalPredictor(xgboost_2_split_1_tree)

        confo_model.calibrate(dmatrix_4x2_with_label)

        mocked_predict_output = np.array([1, 2, 3, 4])

        mocker.patch.object(
            xgb.Booster,
            "predict",
            return_value=mocked_predict_output,
        )

        results = confo_model.predict_with_interval(dmatrix_4x2_with_label)

        np.testing.assert_array_equal(results[:, 1], mocked_predict_output)

    def test_interval_columns_calculation(
        self, mocker, dmatrix_4x2_with_label, xgboost_2_split_1_tree
    ):
        """Test that the interval columns (first and third) are calculated as
        predictions +- baseline_interval.
        """

        confo_model = AbsoluteErrorConformalPredictor(xgboost_2_split_1_tree)

        confo_model.calibrate(dmatrix_4x2_with_label)

        mocked_predict_output = np.array([1, 2, 3, 4])

        mocker.patch.object(
            xgb.Booster,
            "predict",
            return_value=mocked_predict_output,
        )

        results = confo_model.predict_with_interval(dmatrix_4x2_with_label)

        expected_first_column = mocked_predict_output - (confo_model.baseline_interval)

        expected_third_column = mocked_predict_output + (confo_model.baseline_interval)

        np.testing.assert_array_equal(expected_first_column, results[:, 0])

        np.testing.assert_array_equal(expected_third_column, results[:, 2])


class TestCalibrateInterval:
    """Tests for the AbsoluteErrorConformalPredictor._calibrate_interval method."""

    def test_alpha_attribute_set(self, dmatrix_2x1_with_label, xgboost_1_split_1_tree):
        """Test that the alpha attribute is set with the passed value."""

        confo_model = AbsoluteErrorConformalPredictor(xgboost_1_split_1_tree)

        alpha_value = 0.789

        assert not hasattr(
            confo_model, "alpha"
        ), "confo model already has alpha attribute"

        confo_model._calibrate_interval(
            data=dmatrix_2x1_with_label, alpha=alpha_value, response=None
        )

        assert (
            confo_model.alpha == alpha_value
        ), "alpha attribute not set to expected value"

    @pytest.mark.parametrize("use_xgb_dataset_response", [(True), (False)])
    @pytest.mark.parametrize(
        "response, predictions, quantile, expected_baseline_interval",
        [
            (np.array([[1], [1], [1]]), np.array([[1], [2], [3]]), 1, 2),
            (np.array([[1], [1], [1]]), np.array([[1], [2], [-1]]), 1, 2),
            (np.array([[1], [1], [1]]), np.array([[1], [2], [-1]]), 0.5, 1),
            (np.array([[1], [1], [1]]), np.array([[1], [2], [-1]]), 0, 0),
            (
                np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
                np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]),
                0.9,
                9,
            ),
        ],
    )
    def test_baseline_interval_expected_value(
        self,
        mocker,
        xgboost_1_split_1_tree,
        use_xgb_dataset_response,
        response,
        predictions,
        quantile,
        expected_baseline_interval,
    ):
        """Test that baseline_interval is calculated correctly."""

        # set the return value from xgb.Booster.predcit
        mocker.patch.object(xgb.Booster, "predict", return_value=predictions)

        confo_model = AbsoluteErrorConformalPredictor(xgboost_1_split_1_tree)

        if use_xgb_dataset_response:

            # create dummy dataset that will be scored by confo_model.booster
            # the label is set to the value passed in response and will be
            # used when _calibrate_interval is run as response is passed
            # as None
            xgb_dataset = xgb.DMatrix(data=np.ones(response.shape), label=response)

            confo_model._calibrate_interval(
                data=xgb_dataset, alpha=quantile, response=None
            )

        else:

            xgb_dataset = xgb.DMatrix(data=np.ones(response.shape))

            # here, do not set response in the data arg but
            # pass in the response arg directly
            confo_model._calibrate_interval(
                data=xgb_dataset, alpha=quantile, response=response
            )

        assert (
            confo_model.baseline_interval == expected_baseline_interval
        ), "baseline_interval attribute value not correct"
