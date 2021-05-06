import numpy as np
import pandas as pd
import xgboost as xgb
import re

from pitci.xgboost import XGBoosterLeafNodeScaledConformalPredictor
import pitci

import pytest


class TestInit:
    """Tests for the XGBoosterLeafNodeScaledConformalPredictor._init__ method."""

    def test_inheritance(self):
        """Test that XGBoosterLeafNodeScaledConformalPredictor inherits from
        LeafNodeScaledConformalPredictor.
        """

        assert (
            XGBoosterLeafNodeScaledConformalPredictor.__mro__[1]
            is pitci.base.LeafNodeScaledConformalPredictor
        ), (
            "XGBoosterLeafNodeScaledConformalPredictor does not inherit from "
            "LeafNodeScaledConformalPredictor"
        )

    def test_model_type_exception(self):
        """Test an exception is raised if model is not a xgb.Booster object."""

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"model is not in expected types {[xgb.Booster]}, got {tuple}"
            ),
        ):

            XGBoosterLeafNodeScaledConformalPredictor((1, 2, 3))

    def test_attributes_set(self, xgboost_1_split_1_tree):
        """Test that SUPPORTED_OBJECTIVES, version and model attributes are set."""

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        assert (
            confo_model.__version__ == pitci.__version__
        ), "__version__ attribute not set to package version value"

        assert (
            confo_model.model is xgboost_1_split_1_tree
        ), "model attribute not set with the value passed in init"

        assert (
            confo_model.SUPPORTED_OBJECTIVES
            == pitci.xgboost.SUPPORTED_OBJECTIVES_ABS_ERROR
        ), "SUPPORTED_OBJECTIVES attribute incorrect"

    def test_check_objective_supported_called(self, mocker, xgboost_1_split_1_tree):
        """Test that check_objective_supported is called in init."""

        mocked = mocker.patch.object(pitci.xgboost, "check_objective_supported")

        XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        assert (
            mocked.call_count == 1
        ), "check_objective_supported not called (once) in init"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert call_pos_args == (
            xgboost_1_split_1_tree,
            pitci.xgboost.SUPPORTED_OBJECTIVES_ABS_ERROR,
        ), "positional args in check_objective_supported call not correct"

        assert (
            call_kwargs == {}
        ), "keyword args in check_objective_supported call not correct"


class TestCalibrate:
    """Tests for the XGBoosterLeafNodeScaledConformalPredictor.calibrate method."""

    def test_data_type_exception(self, xgboost_1_split_1_tree):
        """Test an exception is raised if data is not a xgb.DMatrix object."""

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[xgb.DMatrix]}, got {str}"
            ),
        ):

            confo_model.calibrate("abcd")

    def test_super_calibrate_call_response_passed(
        self, mocker, dmatrix_2x1_with_label, xgboost_1_split_1_tree
    ):
        """Test XGBoosterLeafNodeScaledConformalPredictor.calibrate call when response is passed."""

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        mocked = mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor, "calibrate"
        )

        response_array = np.array([4, 5])

        confo_model.calibrate(
            data=dmatrix_2x1_with_label, alpha=0.5, response=response_array
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

        assert (
            call_kwargs["data"] == dmatrix_2x1_with_label
        ), "data incorrect in call to LeafNodeScaledConformalPredictor.calibrate"

    def test_super_calibrate_call_no_response_passed(
        self, mocker, dmatrix_2x1_with_label, xgboost_1_split_1_tree
    ):
        """Test LeafNodeScaledConformalPredictor.calibrate call when no response is passed."""

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        mocked = mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor, "calibrate"
        )

        confo_model.calibrate(data=dmatrix_2x1_with_label, alpha=0.99)

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
            call_kwargs["alpha"] == 0.99
        ), "alpha incorrect in call to LeafNodeScaledConformalPredictor.calibrate"

        np.testing.assert_array_equal(
            call_kwargs["response"], dmatrix_2x1_with_label.get_label()
        )

        assert (
            call_kwargs["data"] == dmatrix_2x1_with_label
        ), "data incorrect in call to LeafNodeScaledConformalPredictor.calibrate"


class TestPredictWithInterval:
    """Tests for the XGBoosterLeafNodeScaledConformalPredictor.predict_with_interval method."""

    def test_data_type_exception(self, dmatrix_2x1_with_label, xgboost_1_split_1_tree):
        """Test an exception is raised if data is not a xgb.DMatrix object."""

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        confo_model.calibrate(dmatrix_2x1_with_label)

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[xgb.DMatrix]}, got {pd.DataFrame}"
            ),
        ):

            confo_model.predict_with_interval(pd.DataFrame())

    def test_no_leaf_node_counts_attribute_exception(
        self, dmatrix_2x1_with_label, xgboost_1_split_1_tree
    ):
        """Test an exception is raised if leaf_node_counts attribute is not present."""

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        assert not hasattr(
            confo_model, "leaf_node_counts"
        ), "XGBoosterLeafNodeScaledConformalPredictor has leaf_node_counts attribute prior to running calibrate"

        with pytest.raises(
            AttributeError,
            match="XGBoosterLeafNodeScaledConformalPredictor does not have leaf_node_counts"
            " attribute, run calibrate first.",
        ):

            confo_model.predict_with_interval(dmatrix_2x1_with_label)

    def test_super_predict_with_interval_call(
        self, mocker, dmatrix_2x1_with_label, xgboost_1_split_1_tree
    ):
        """Test that LeafNodeScaledConformalPredictor.predict_with_interval is called and the
        outputs of this are returned from the method.
        """

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        confo_model.calibrate(dmatrix_2x1_with_label)

        predict_return_value = np.array([200, 101, 1234])

        mocked = mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor,
            "predict_with_interval",
            return_value=predict_return_value,
        )

        results = confo_model.predict_with_interval(dmatrix_2x1_with_label)

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

        assert call_kwargs == {
            "data": dmatrix_2x1_with_label
        }, "keyword args incorrect in call to LeafNodeScaledConformalPredictor.predict_with_interval"


class TestGeneratePredictions:
    """Tests for the XGBoosterLeafNodeScaledConformalPredictor._generate_predictions method."""

    def test_data_type_exception(self, dmatrix_2x1_with_label, xgboost_1_split_1_tree):
        """Test an exception is raised if data is not a xgb.DMatrix object."""

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[xgb.DMatrix]}, got {float}"
            ),
        ):

            confo_model._generate_predictions(12345.0)

    def test_predict_call(self, mocker, dmatrix_2x1_with_label, xgboost_1_split_1_tree):
        """Test that the output from xgb.Booster.predict with ntree_limit = best_iteration + 1
        is returned from the method.
        """

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        confo_model.calibrate(dmatrix_2x1_with_label)

        predict_return_value = np.array([200, 101])

        mocked = mocker.patch.object(
            xgb.Booster, "predict", return_value=predict_return_value
        )

        results = confo_model._generate_predictions(dmatrix_2x1_with_label)

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to xgb.Booster.predict"

        np.testing.assert_array_equal(results, predict_return_value)

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert call_pos_args == (
            dmatrix_2x1_with_label,
        ), "positional args incorrect in call to xgb.Booster.predict"

        assert call_kwargs == {
            "ntree_limit": xgboost_1_split_1_tree.best_iteration + 1
        }, "positional args incorrect in call to xgb.Booster.predict"


class TestGenerateLeafNodePredictions:
    """Tests for the XGBoosterLeafNodeScaledConformalPredictor._generate_leaf_node_predictions
    method.
    """

    def test_data_type_exception(self, dmatrix_2x1_with_label, xgboost_1_split_1_tree):
        """Test an exception is raised if data is not a xgb.DMatrix object."""

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[xgb.DMatrix]}, got {list}"
            ),
        ):

            confo_model._generate_leaf_node_predictions([])

    def test_predict_call(self, mocker, dmatrix_2x1_with_label, xgboost_1_split_1_tree):
        """Test that the output from xgb.Booster.predict with ntree_limit = best_iteration + 1
        and pred_leaf = True is returned from the method.
        """

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        confo_model.calibrate(dmatrix_2x1_with_label)

        predict_return_value = np.array([[200, 101], [5, 6]])

        mocked = mocker.patch.object(
            xgb.Booster, "predict", return_value=predict_return_value
        )

        results = confo_model._generate_leaf_node_predictions(dmatrix_2x1_with_label)

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to xgb.Booster.predict"

        np.testing.assert_array_equal(results, predict_return_value)

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            call_pos_args == ()
        ), "positional args incorrect in call to xgb.Booster.predict"

        assert call_kwargs == {
            "ntree_limit": xgboost_1_split_1_tree.best_iteration + 1,
            "data": dmatrix_2x1_with_label,
            "pred_leaf": True,
        }, "positional args incorrect in call to xgb.Booster.predict"

    def test_output_2d(self, mocker, dmatrix_2x1_with_label, xgboost_1_split_1_tree):
        """Test the array returned from _generate_leaf_node_predictions is a 2d array
        even if the output from predict is 1d.
        """

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_1_split_1_tree)

        confo_model.calibrate(dmatrix_2x1_with_label)

        # set the return value from predict to be a 1d array
        predict_return_value = np.array([200, 101])

        mocker.patch.object(xgb.Booster, "predict", return_value=predict_return_value)

        results = confo_model._generate_leaf_node_predictions(dmatrix_2x1_with_label)

        expected_results = predict_return_value.reshape(
            predict_return_value.shape[0], 1
        )

        np.testing.assert_array_equal(results, expected_results)


class TestCalibrateLeafNodeCounts:
    """Tests that _calibrate_leaf_node_counts calculate the correct values."""

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

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_2_split_1_tree)

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

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_2_split_1_tree)

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

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_2_split_2_tree)

        confo_model._calibrate_leaf_node_counts(xgb_data)

        assert (
            confo_model.leaf_node_counts == expected_leaf_node_counts
        ), "leaf_node_counts not calculated correctly"

    def test_leaf_node_counts_excludes_non_visited_nodes(self, xgboost_2_split_2_tree):
        """Test that leaf_node_counts does not include nodes that were not visited when
        predicting on data.
        """

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
        expected_leaf_node_counts = [{2: 8}, {1: 4, 2: 4}]

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(xgboost_2_split_2_tree)

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

        confo_model = XGBoosterLeafNodeScaledConformalPredictor(model)

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

            assert (
                confo_model.leaf_node_counts[column_no] == counts
            ), f"incorrect leaf node count for tree {column_no}"