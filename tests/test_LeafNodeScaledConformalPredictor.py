import numpy as np
import pandas as pd
import xgboost as xgb
import re

from pitci import LeafNodeScaledConformalPredictor
import pitci.intervals as intervals
import pitci

import pytest


class TestInit:
    """Tests for the LeafNodeScaledConformalPredictor._init__ method."""

    def test_attributes_set(self, xgboost_1_split_1_tree):
        """Test that version and booster attributes are set."""

        confo_model = LeafNodeScaledConformalPredictor(
            xgboost_1_split_1_tree, scaling_function="leaf_node_counts"
        )

        assert (
            confo_model.scaling_function == "leaf_node_counts"
        ), "scaling_function attribute not set to value passed in init"

    def test_super_init_called(self, mocker, xgboost_1_split_1_tree):
        """Test that AbsoluteErrorConformalPredictor.__init__ is called."""

        mocked = mocker.patch.object(
            pitci.intervals.AbsoluteErrorConformalPredictor, "__init__"
        )

        LeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to AbsoluteErrorConformalPredictor.__init__"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            call_pos_args == ()
        ), "incorrect positional args in AbsoluteErrorConformalPredictor.__init__ call"

        assert call_kwargs == {
            "booster": xgboost_1_split_1_tree
        }, "incorrect kwargs in AbsoluteErrorConformalPredictor.__init__ call"


class TestCalibrate:
    """Tests for the LeafNodeScaledConformalPredictor.calibrate method."""

    @pytest.mark.parametrize("alpha", [(-0.0001), (-1), (1.0001), (2), (55)])
    def test_alpha_value_error(
        self, dmatrix_2x1_with_label, xgboost_1_split_1_tree, alpha
    ):
        """Test an exception is raised if alpha is below 0 or greater than 1."""

        confo_model = LeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        with pytest.raises(
            ValueError, match=re.escape("alpha must be in range [0 ,1]")
        ):

            confo_model.calibrate(data=dmatrix_2x1_with_label, alpha=alpha)

    @pytest.mark.parametrize(
        "attribute_name",
        [("leaf_node_counts"), ("alpha"), ("baseline_interval")],
    )
    def test_attributes_set(
        self, dmatrix_2x1_with_label, xgboost_1_split_1_tree, attribute_name
    ):
        """Test attributes are set after running method."""

        confo_model = LeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        assert not hasattr(
            confo_model, attribute_name
        ), f"LeafNodeScaledConformalPredictor already has {attribute_name} before running set_attributes"

        confo_model.calibrate(dmatrix_2x1_with_label)

        assert hasattr(
            confo_model, attribute_name
        ), f"LeafNodeScaledConformalPredictor does not have {attribute_name} after running set_attributes"


class TestCalibrateInterval:
    """Tests for the LeafNodeScaledConformalPredictor._calibrate_interval method."""

    def test_alpha_attribute_set(self, dmatrix_2x1_with_label, xgboost_1_split_1_tree):
        """Test that the alpha attribute is set with the passed value."""

        confo_model = LeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        # run _calibrate_leaf_node_counts to set attributes
        confo_model._calibrate_leaf_node_counts(data=dmatrix_2x1_with_label)

        alpha_value = 0.4567

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

        confo_model = LeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        if use_xgb_dataset_response:

            # create dummy dataset that will be scored by confo_model.booster
            # the label is set to the value passed in response and will be
            # used when _calibrate_interval is run as response is passed
            # as None
            xgb_dataset = xgb.DMatrix(data=np.ones(response.shape), label=response)

            # run _calibrate_leaf_node_counts to set attributes
            confo_model._calibrate_leaf_node_counts(data=xgb_dataset)

            confo_model._calibrate_interval(
                data=xgb_dataset, alpha=quantile, response=None
            )

        else:

            xgb_dataset = xgb.DMatrix(data=np.ones(response.shape))

            # run _calibrate_leaf_node_counts to set attributes
            confo_model._calibrate_leaf_node_counts(data=xgb_dataset)

            # here, do not set response in the data arg but
            # pass in the response arg directly
            confo_model._calibrate_interval(
                data=xgb_dataset, alpha=quantile, response=response
            )

        assert (
            confo_model.baseline_interval == expected_baseline_interval
        ), "baseline_interval attribute value not correct"


class TestSumDictValues:
    """Tests for the intervals._sum_dict_values function."""

    @pytest.mark.parametrize(
        "arr, counts, expected_output",
        [
            (np.array([1]), {0: {1: 123}}, 123),
            (
                np.array([1, 1, 1]),
                {0: {1: 123, 0: 21}, 1: {3: -1, 1: 100}, 2: {1: 5}},
                228,
            ),
            (
                np.array([1, 2, 3]),
                {0: {1: -1}, 1: {3: 21, 1: 100, 2: -1}, 2: {1: 5, 2: 99, 3: -1}},
                -3,
            ),
        ],
    )
    def test_expected_output(self, arr, counts, expected_output):
        """Test the correct values are summed in function."""

        output = intervals._sum_dict_values(arr, counts)

        assert output == expected_output, "_sum_dict_values produced incorrect output"


class TestCountLeafNodeVisitsFromCalibration:
    """Tests for the LeafNodeScaledConformalPredictor._count_leaf_node_visits_from_calibration method."""

    def test_sum_dict_values_calls(self, mocker, xgboost_1_split_1_tree):
        """Test that _sum_dict_values is called as expected."""

        spy = mocker.spy(pitci.intervals, "_sum_dict_values")

        confo_model = LeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        leaf_nodes_row_1 = np.array([[1, 2, 3, 4]])
        leaf_nodes_row_2 = np.array([[5, 6, 7, 8]])
        leaf_nodes_row_3 = np.array([[9, 10, 11, 12]])

        leaf_node_predictions_array = np.concatenate(
            [leaf_nodes_row_1, leaf_nodes_row_2, leaf_nodes_row_3], axis=0
        )

        confo_model.leaf_node_counts = {
            0: {1: 1, 5: 11, 9: 4},
            1: {2: -1, 6: 99, 10: 20},
            2: {3: 5, 7: 33, 11: 14},
            3: {4: -9, 8: 77, 12: -2},
        }

        confo_model._count_leaf_node_visits_from_calibration(
            leaf_node_predictions_array
        )

        assert spy.call_count == 3, "_sum_dict_values called incorrect number of times"

        array_row_list = [leaf_nodes_row_1, leaf_nodes_row_2, leaf_nodes_row_3]

        for i, array_row in enumerate(array_row_list):

            call_i_args = spy.call_args_list[i]
            call_i_pos_args = call_i_args[0]
            call_i_kwargs = call_i_args[1]

            assert call_i_kwargs == {"counts": confo_model.leaf_node_counts}

            assert (
                len(call_i_pos_args) == 1
            ), f"incorrect number of args in _sum_dict_values call {i}"

            np.testing.assert_array_equal(call_i_pos_args[0], array_row.reshape(-1))


class TestPredictWithInterval:
    """Tests for the LeafNodeScaledConformalPredictor.predict_with_interval method."""

    @pytest.mark.parametrize("n_input_rows", [(1), (2), (5), (100)])
    def test_output_shape_single_column_input(
        self, dmatrix_2x1_with_label, xgboost_1_split_1_tree, n_input_rows
    ):
        """Test the output from the function has the same number of rows as the input and 3 columns."""

        confo_model = LeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

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

        confo_model = LeafNodeScaledConformalPredictor(xgboost_2_split_1_tree)

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

        confo_model = LeafNodeScaledConformalPredictor(xgboost_2_split_1_tree)

        confo_model.calibrate(dmatrix_4x2_with_label)

        mocked_first_output = np.array([1, 2, 3, 4])

        # note, the second output will be used in counting leaf nodes
        # so values have to be leaf node indexes
        mocked_second_output = np.array([1, 1, 4, 3])

        mocker.patch.object(
            xgb.Booster,
            "predict",
            side_effect=[mocked_first_output, mocked_second_output],
        )

        results = confo_model.predict_with_interval(dmatrix_4x2_with_label)

        np.testing.assert_array_equal(results[:, 1], mocked_first_output)

    def test_interval_columns_calculation(
        self, mocker, dmatrix_4x2_with_label, xgboost_2_split_1_tree
    ):
        """Test that the interval columns (first and third) are calculated as
        predictions +- (baseline_interval / (_count_leaf_node_visits_from_calibration output / median_normaliser)).
        """

        confo_model = LeafNodeScaledConformalPredictor(xgboost_2_split_1_tree)

        confo_model.calibrate(dmatrix_4x2_with_label)

        predict_first_output = np.array([1, 2, 3, 4])

        # note, the second output will be used in counting leaf nodes
        # so values have to be leaf node indexes
        predict_second_output = np.array([1, 1, 4, 3])

        mocker.patch.object(
            xgb.Booster,
            "predict",
            side_effect=[predict_first_output, predict_second_output],
        )

        count_leaf_nodes_output = np.array([8, 9, -1, 55])

        mocker.patch.object(
            pitci.LeafNodeScaledConformalPredictor,
            "_count_leaf_node_visits_from_calibration",
            return_value=count_leaf_nodes_output,
        )

        results = confo_model.predict_with_interval(dmatrix_4x2_with_label)

        expected_first_column = predict_first_output - (
            confo_model.baseline_interval * count_leaf_nodes_output
        )

        expected_third_column = predict_first_output + (
            confo_model.baseline_interval * count_leaf_nodes_output
        )

        np.testing.assert_array_equal(expected_first_column, results[:, 0])

        np.testing.assert_array_equal(expected_third_column, results[:, 2])


class TestCalibrateLeafNodeCounts:
    """Tests for the LeafNodeScaledConformalPredictor._calibrate_leaf_node_counts method."""

    def test_leaf_node_counts_correct_1(
        self, xgboost_2_split_1_tree, dmatrix_4x2_with_label
    ):
        """Test the leaf_node_counts attribute has the correct values with hand workable example."""

        # rules for xgboost_2_split_1_tree are as follows;
        # leaf 1 - if (f0 < 0.5)
        # leaf 3 - if (f0 > 0.5) & (f1 < 0.5)
        # leaf 4 - if (f0 > 0.5) & (f1 > 0.5)

        # there for the dmatrix_4x2_with_label data will be mapped to;
        # [1, 1] - leaf 4
        # [1, 0] - leaf 3
        # [0, 1] - leaf 1
        # [0, 0] - leaf 1

        # therefore the leaf_node_counts attribute for a single tree
        # should be;
        expected_leaf_node_counts = [{1: 2, 3: 1, 4: 1}]

        confo_model = LeafNodeScaledConformalPredictor(xgboost_2_split_1_tree)

        confo_model._calibrate_leaf_node_counts(dmatrix_4x2_with_label)

        assert (
            confo_model.leaf_node_counts == expected_leaf_node_counts
        ), "leaf_node_counts not calculated correctly"

    def test_leaf_node_counts_correct_2(self, xgboost_2_split_1_tree):
        """Test the leaf_node_counts attribute has the correct values with 2nd hand workable example."""

        # rules for xgboost_2_split_1_tree are as follows;
        # leaf 1 - if (f0 < 0.5)
        # leaf 3 - if (f0 > 0.5) & (f1 < 0.5)
        # leaf 4 - if (f0 > 0.5) & (f1 > 0.5)

        # for this dataset the leaf nodes for each row are inline below;
        xgb_data = xgb.DMatrix(
            data=np.array(
                [
                    [1, 1],  # leaf 4
                    [1, 0],  # leaf 3
                    [0, 1],  # leaf 1
                    [0, 0],  # leaf 1
                    [1, 0],  # leaf 3
                    [0, 1],  # leaf 1
                    [0, 0],  # leaf 1
                ]
            )
        )

        # therefore the leaf_node_counts attribute for a single tree
        # should be;
        expected_leaf_node_counts = [{1: 4, 3: 2, 4: 1}]

        confo_model = LeafNodeScaledConformalPredictor(xgboost_2_split_1_tree)

        confo_model._calibrate_leaf_node_counts(xgb_data)

        assert (
            confo_model.leaf_node_counts == expected_leaf_node_counts
        ), "leaf_node_counts not calculated correctly"

    def test_leaf_node_counts_correct_3(self, xgboost_2_split_2_tree):
        """Test the leaf_node_counts attribute has the correct values with 3rd hand workable example."""

        # rules for xgboost_2_split_1_tree are as follows;
        # tree 1, leaf 1 - if (f0 < 0.5)
        # tree 1, leaf 2 - if (f0 > 0.5)
        # tree 2, leaf 1 - if (f1 < 0.5)
        # tree 2, leaf 2 - if (f1 > 0.5)

        # for this dataset the leaf nodes for each row are inline below;
        xgb_data = xgb.DMatrix(
            data=np.array(
                [
                    [1, 1],  # tree 1, leaf 2 > tree 2, leaf 2
                    [1, 1],  # tree 1, leaf 2 > tree 2, leaf 2
                    [1, 1],  # tree 1, leaf 2 > tree 2, leaf 2
                    [1, 1],  # tree 1, leaf 2 > tree 2, leaf 2
                    [1, 0],  # tree 1, leaf 2 > tree 2, leaf 1
                    [1, 0],  # tree 1, leaf 2 > tree 2, leaf 1
                    [1, 0],  # tree 1, leaf 2 > tree 2, leaf 1
                    [1, 0],  # tree 1, leaf 2 > tree 2, leaf 1
                    [0, 1],  # tree 1, leaf 1 > tree 2, leaf 2
                ]
            )
        )

        # therefore the leaf_node_counts attribute for a single tree
        # should be;
        expected_leaf_node_counts = [{1: 1, 2: 8}, {1: 4, 2: 5}]

        confo_model = LeafNodeScaledConformalPredictor(xgboost_2_split_2_tree)

        confo_model._calibrate_leaf_node_counts(xgb_data)

        assert (
            confo_model.leaf_node_counts == expected_leaf_node_counts
        ), "leaf_node_counts not calculated correctly"

    def test_leaf_node_counts_includes_non_visited_nodes(self, xgboost_2_split_2_tree):
        """Test that leaf_node_counts includes nodes that were not visited when predicting on
        data."""

        # note, this dataset does not include any rows that visit tree 1, leaf 1
        # for this dataset the leaf nodes for each row are inline below;
        xgb_data = xgb.DMatrix(
            data=np.array(
                [
                    [1, 1],  # tree 1, leaf 2 > tree 2, leaf 2
                    [1, 1],  # tree 1, leaf 2 > tree 2, leaf 2
                    [1, 1],  # tree 1, leaf 2 > tree 2, leaf 2
                    [1, 1],  # tree 1, leaf 2 > tree 2, leaf 2
                    [1, 0],  # tree 1, leaf 2 > tree 2, leaf 1
                    [1, 0],  # tree 1, leaf 2 > tree 2, leaf 1
                    [1, 0],  # tree 1, leaf 2 > tree 2, leaf 1
                    [1, 0],  # tree 1, leaf 2 > tree 2, leaf 1
                ]
            )
        )

        # therefore the leaf_node_counts attribute for a single tree
        # should be;
        expected_leaf_node_counts = [{1: 0, 2: 8}, {1: 4, 2: 4}]

        confo_model = LeafNodeScaledConformalPredictor(xgboost_2_split_2_tree)

        confo_model._calibrate_leaf_node_counts(xgb_data)

        assert (
            confo_model.leaf_node_counts == expected_leaf_node_counts
        ), "leaf_node_counts not calculated correctly when nodes not visited"

    @pytest.mark.parametrize("dataset", [("diabetes_xgb_data")])
    @pytest.mark.parametrize(
        "params_dict",
        [
            ({"max_depth": 5, "eta": 0.09}),
            ({"max_depth": 3, "eta": 0.09}),
            ({"max_depth": 1, "eta": 0.09}),
            ({"max_depth": 3, "eta": 0.11}),
            ({"max_depth": 7, "eta": 0.05}),
        ],
    )
    def test_test_leaf_node_counts_correct(self, dataset, params_dict, request):
        """Test leaf_node_counts is calculated correctly - on larger models that require automated
        calculation to check against what is produced by _calibrate_leaf_node_counts.
        """

        # this is used to parameterise which fixture to use to provide the data
        dataset = request.getfixturevalue(dataset)

        # build model with params passed
        model = xgb.train(
            params=params_dict,
            dtrain=dataset[0],
            num_boost_round=500,
            evals=[(dataset[1], "validate")],
            early_stopping_rounds=5,
            verbose_eval=False,
        )

        model_df = model.trees_to_dataframe()

        confo_model = LeafNodeScaledConformalPredictor(model)

        # set leaf_node_counts attribute
        # note, we are using a different dataset to training so not guaranteed to have
        # every leaf node in the model visited
        confo_model._calibrate_leaf_node_counts(dataset[2])

        # now calculate values (leaf node counts) from scratch
        # first generate leaf node predictions
        leaf_node_predictions = model.predict(
            data=dataset[2], pred_leaf=True, ntree_limit=model.best_iteration + 1
        )

        # loop through each column i.e. tree
        for column_no in range(leaf_node_predictions.shape[1]):

            # these are the counts we expected to see in confo_model.leaf_node_counts[column_no]
            # unless a particular node was not visited at all in the dataset
            counts = (
                pd.Series(leaf_node_predictions[:, column_no]).value_counts().to_dict()
            )

            # get all the leaf nodes for that tree, from the export of the model to df
            tree_leaf_nodes = model_df.loc[
                (model_df["Tree"] == column_no) & (model_df["Feature"] == "Leaf"),
                "Node",
            ].tolist()

            for tree_leaf_node in tree_leaf_nodes:

                if tree_leaf_node in counts.keys():

                    expected_count_value = counts[tree_leaf_node]

                else:

                    expected_count_value = 0

                assert (
                    confo_model.leaf_node_counts[column_no][tree_leaf_node]
                    == expected_count_value
                ), f"incorrect leaf node count for node {tree_leaf_node} in tree {column_no}"
