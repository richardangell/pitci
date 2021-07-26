import numpy as np
import pandas as pd
import lightgbm as lgb
import re

from pitci.lightgbm import LGBMBoosterLeafNodeScaledConformalPredictor
import pitci

import pytest


class TestInit:
    """Tests for the LGBMBoosterLeafNodeScaledConformalPredictor._init__ method."""

    def test_inheritance(self):
        """Test that LGBMBoosterLeafNodeScaledConformalPredictor inherits from
        LeafNodeScaledConformalPredictor.
        """

        assert (
            LGBMBoosterLeafNodeScaledConformalPredictor.__mro__[1]
            is pitci.base.LeafNodeScaledConformalPredictor
        ), (
            "LGBMBoosterLeafNodeScaledConformalPredictor does not inherit from "
            "LeafNodeScaledConformalPredictor"
        )

    def test_model_type_exception(self):
        """Test an exception is raised if model is not a lgb.Booster object."""

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"model is not in expected types {[lgb.basic.Booster]}, got {list}"
            ),
        ):

            LGBMBoosterLeafNodeScaledConformalPredictor([1, 2, 3])

    def test_attributes_set(self, lgb_booster_1_split_1_tree):
        """Test that SUPPORTED_OBJECTIVES, version and model attributes are set."""

        confo_model = LGBMBoosterLeafNodeScaledConformalPredictor(
            lgb_booster_1_split_1_tree
        )

        assert (
            confo_model.__version__ == pitci.__version__
        ), "__version__ attribute not set to package version value"

        assert (
            confo_model.model is lgb_booster_1_split_1_tree
        ), "model attribute not set with the value passed in init"

        assert (
            confo_model.SUPPORTED_OBJECTIVES
            == pitci.lightgbm.SUPPORTED_OBJECTIVES_ABS_ERROR
        ), "SUPPORTED_OBJECTIVES attribute incorrect"

    def test_check_objective_supported_called(self, mocker, lgb_booster_1_split_1_tree):
        """Test that check_objective_supported is called in init."""

        mocked = mocker.patch.object(pitci.lightgbm, "check_objective_supported")

        LGBMBoosterLeafNodeScaledConformalPredictor(lgb_booster_1_split_1_tree)

        assert (
            mocked.call_count == 1
        ), "check_objective_supported not called (once) in init"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert call_pos_args == (
            lgb_booster_1_split_1_tree,
            pitci.lightgbm.SUPPORTED_OBJECTIVES_ABS_ERROR,
        ), "positional args in check_objective_supported call not correct"

        assert (
            call_kwargs == {}
        ), "keyword args in check_objective_supported call not correct"

    def test_super_init_call(self, mocker, lgb_booster_1_split_1_tree):
        """Test that LeafNodeScaledConformalPredictor.__init__ is called."""

        mocked = mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor, "__init__"
        )

        LGBMBoosterLeafNodeScaledConformalPredictor(lgb_booster_1_split_1_tree)

        assert (
            mocked.call_count == 1
        ), "LeafNodeScaledConformalPredictor.__init__ not called (once) in init"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            call_pos_args == ()
        ), "positional args in LeafNodeScaledConformalPredictor.__init__ call not correct"

        assert call_kwargs == {
            "model": lgb_booster_1_split_1_tree
        }, "keyword args in LeafNodeScaledConformalPredictor.__init__ call not correct"


class TestCalibrate:
    """Tests for the LGBMBoosterLeafNodeScaledConformalPredictor.calibrate method."""

    def test_data_exception(self, lgb_booster_1_split_1_tree):
        """Test that an exception is raised if data is not a np.array or pd.Series."""

        confo_model = LGBMBoosterLeafNodeScaledConformalPredictor(
            lgb_booster_1_split_1_tree
        )

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[np.ndarray, pd.DataFrame]}, got {int}"
            ),
        ):

            confo_model.calibrate(data=1, response=np.ndarray([1, 2]), alpha=0.8)

    def test_train_data_type_exception(self, lgb_booster_1_split_1_tree):
        """Test that an exception is raised if train_data is not a np.array or pd.Series."""

        confo_model = LGBMBoosterLeafNodeScaledConformalPredictor(
            lgb_booster_1_split_1_tree
        )

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"train_data is not in expected types {[np.ndarray, pd.DataFrame, type(None)]}, got {bool}"
            ),
        ):

            confo_model.calibrate(
                data=np.ndarray([1, 2]), response=np.ndarray([1, 2]), train_data=True
            )

    def test_super_calibrate_call(
        self, mocker, np_2x1_with_label, lgb_booster_1_split_1_tree
    ):
        """Test that LeafNodeScaledConformalPredictor.calibrate is called correctly."""

        confo_model = LGBMBoosterLeafNodeScaledConformalPredictor(
            lgb_booster_1_split_1_tree
        )

        mocked = mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor, "calibrate"
        )

        train_data_array = np.array([6, 9])

        confo_model.calibrate(
            data=np_2x1_with_label[0],
            alpha=0.99,
            response=np_2x1_with_label[1],
            train_data=train_data_array,
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

        assert call_kwargs["alpha"] == 0.99

        np.testing.assert_array_equal(call_kwargs["data"], np_2x1_with_label[0])

        np.testing.assert_array_equal(call_kwargs["response"], np_2x1_with_label[1])

        np.testing.assert_array_equal(call_kwargs["train_data"], train_data_array)


class TestPredictWithInterval:
    """Tests for the LGBMBoosterLeafNodeScaledConformalPredictor.predict_with_interval method."""

    def test_data_type_exception(self, lgb_booster_1_split_1_tree):
        """Test an exception is raised if data is not a xgb.DMatrix object."""

        confo_model = LGBMBoosterLeafNodeScaledConformalPredictor(
            lgb_booster_1_split_1_tree
        )

        confo_model.calibrate(
            data=np.array([[1], [2], [3]]),
            response=np.array([1, 2, 3]),
        )

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"data is not in expected types {[np.ndarray, pd.DataFrame]}, got {int}"
            ),
        ):

            confo_model.predict_with_interval(1)

    def test_super_predict_with_interval_call(
        self, mocker, dmatrix_2x1_with_label, lgb_booster_1_split_1_tree
    ):
        """Test that LeafNodeScaledConformalPredictor.predict_with_interval is called and the
        outputs of this are returned from the method.
        """

        confo_model = LGBMBoosterLeafNodeScaledConformalPredictor(
            lgb_booster_1_split_1_tree
        )

        confo_model.calibrate(np.array([[1], [2], [3]]), response=np.array([1, 2, 3]))

        predict_return_value = np.array([200, 101, 1234])

        mocked = mocker.patch.object(
            pitci.base.LeafNodeScaledConformalPredictor,
            "predict_with_interval",
            return_value=predict_return_value,
        )

        data_array = np.array([[11], [21], [31]])

        results = confo_model.predict_with_interval(data_array)

        # test output of predict_with_interval is the return value of
        # LeafNodeScaledConformalPredictor.predict_with_interval
        np.testing.assert_array_equal(results, predict_return_value)

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to LeafNodeScaledConformalPredictor.predict_with_interval"

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            call_pos_args == ()
        ), "positional args incorrect in call to LeafNodeScaledConformalPredictor.predict_with_interval"

        np.testing.assert_array_equal(call_kwargs["data"], data_array)


class TestCalibrateLeafNodeCounts:
    """Tests for the LGBMBoosterLeafNodeScaledConformalPredictor._calibrate_leaf_node_counts method."""

    def test_leaf_node_counts_correct(self, diabetes_lgb_data):
        """The the leaf_node_counts attribute is set correctly."""

        model = lgb.train(
            params={
                "num_leaves": 31,
                "learning_rate": 0.05,
                "metric": "mean_squared_error",
                "verbosity": -1,
            },
            train_set=diabetes_lgb_data[0],
            num_boost_round=500,
            valid_sets=[diabetes_lgb_data[1]],
            valid_names=["validate"],
            early_stopping_rounds=5,
            verbose_eval=0,
        )

        confo_model = LGBMBoosterLeafNodeScaledConformalPredictor(model)

        confo_model._calibrate_leaf_node_counts("abcd")

        assert hasattr(
            confo_model, "leaf_node_counts"
        ), "leaf_node_counts attribute not set in _calibrate_leaf_node_counts"

        assert (
            type(confo_model.leaf_node_counts) is list
        ), "leaf_node_counts is not a list"

        assert (
            len(confo_model.leaf_node_counts) == model.best_iteration
        ), "length of leaf_node_counts attribute incorrect"

        trees_df = model.trees_to_dataframe()

        for tree_no, tree_leaf_node_counts in enumerate(confo_model.leaf_node_counts):

            assert (
                type(tree_leaf_node_counts) is dict
            ), f"incorrect type for {tree_no}th leaf_node_count"

            for leaf_index, leaf_index_counts in tree_leaf_node_counts.items():

                node_index = f"{tree_no}-L{leaf_index}"

                assert (
                    trees_df.loc[trees_df["node_index"] == node_index, "count"].values[
                        0
                    ]
                    == leaf_index_counts
                ), f"incorrect leaf node counts for tree number {tree_no} and leaf index {leaf_index}"


class TestGeneratePredictions:
    """Tests for the LGBMBoosterLeafNodeScaledConformalPredictor._generate_predictions method."""

    def test_lgb_booster_predict_call(
        self, mocker, np_2x1_with_label, lgb_booster_1_split_1_tree
    ):
        """Test that lgb.Booster.predict is called and the output is returned
        from _generate_predictions.
        """

        confo_model = LGBMBoosterLeafNodeScaledConformalPredictor(
            lgb_booster_1_split_1_tree
        )

        confo_model.calibrate(np_2x1_with_label[0], np_2x1_with_label[1])

        predict_return_value = np.array([200, 101])

        mocked = mocker.patch.object(
            lgb.basic.Booster, "predict", return_value=predict_return_value
        )

        results = confo_model._generate_predictions(np_2x1_with_label[0])

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to lgb.basic.Booster.predict"

        np.testing.assert_array_equal(results, predict_return_value)

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            len(call_pos_args) == 1
        ), "incorrect number of positional args incorrect in call to lgb.Booster.predict"

        np.testing.assert_array_equal(call_pos_args[0], np_2x1_with_label[0])

        assert (
            call_kwargs == {}
        ), "keyword args incorrect in call to xgb.Booster.predict"


class TestGenerateLeafNodePredictions:
    """Tests for the LGBMBoosterLeafNodeScaledConformalPredictor._generate_leaf_node_predictions methods."""

    def test_predict_call(self, mocker, np_2x1_with_label, lgb_booster_1_split_1_tree):
        """Test that the output from lgb.Booster.predict (with pred_leaf set to True)
        is returned from the method.
        """

        confo_model = LGBMBoosterLeafNodeScaledConformalPredictor(
            lgb_booster_1_split_1_tree
        )

        confo_model.calibrate(np_2x1_with_label[0], np_2x1_with_label[1])

        predict_return_value = np.array([[200, 101], [5, 6]])

        mocked = mocker.patch.object(
            lgb.basic.Booster, "predict", return_value=predict_return_value
        )

        results = confo_model._generate_leaf_node_predictions(np_2x1_with_label[0])

        assert (
            mocked.call_count == 1
        ), "incorrect number of calls to lgb.basic.Booster.predict"

        np.testing.assert_array_equal(results, predict_return_value)

        call_args = mocked.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        assert (
            len(call_pos_args) == 1
        ), "incorrect number of positional args incorrect in call to lgb.Booster.predict"

        np.testing.assert_array_equal(call_pos_args[0], np_2x1_with_label[0])

        assert call_kwargs == {
            "pred_leaf": True
        }, "keyword args incorrect in call to xgb.Booster.predict"

    def test_output_2d(self, mocker, np_2x1_with_label, lgb_booster_1_split_1_tree):
        """Test the array returned from _generate_leaf_node_predictions is a 2d array
        even if the output from predict is 1d.
        """

        confo_model = LGBMBoosterLeafNodeScaledConformalPredictor(
            lgb_booster_1_split_1_tree
        )

        confo_model.calibrate(np_2x1_with_label[0], np_2x1_with_label[1])

        predict_return_value = np.array([200, 101])

        mocker.patch.object(
            lgb.basic.Booster, "predict", return_value=predict_return_value
        )

        results = confo_model._generate_leaf_node_predictions(np_2x1_with_label[0])

        expected_results = predict_return_value.reshape(
            predict_return_value.shape[0], 1
        )

        np.testing.assert_array_equal(results, expected_results)
