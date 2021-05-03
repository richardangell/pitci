import xgboost as xgb
import re
import pytest

import pitci.xgboost as pitci_xgb


class TestCheckObjectiveSupported:
    """Tests for the check_objective_supported function."""

    @pytest.mark.parametrize(
        "objective, supported_objectives, message",
        [
            ("reg:squarederror", ["reg:gamma", "reg:tweedie"], "abcde"),
            ("reg:logistic", ["reg:pseudohubererror", "count:poisson"], ""),
        ],
    )
    def test_exception_raised(
        self, dmatrix_2x1_with_label, objective, supported_objectives, message
    ):
        """Test an exception is raised if a model with an object not in the
        supported_objective list.
        """

        params = {"objective": objective}

        model = xgb.train(
            params=params, dtrain=dmatrix_2x1_with_label, num_boost_round=1
        )

        error_message = f"booster objective not supported\n{objective} not in allowed values; {supported_objectives}"

        with pytest.raises(
            ValueError,
            match=re.escape(error_message),
        ):

            pitci_xgb.check_objective_supported(model, supported_objectives)


class TestDispatchingFunctions:
    """Tests for the functions being dispatched to from the dispatchers module."""

    def test_return_xgb_booster_absolute_error_confromal_predictor(
        self, xgboost_1_split_1_tree
    ):
        """Test return_xgb_booster_absolute_error_confromal_predictor function returns
        an XGBoosterAbsoluteErrorConformalPredictor object with model passed.
        """

        confo_model = pitci_xgb.return_xgb_booster_absolute_error_confromal_predictor(
            xgboost_1_split_1_tree
        )

        assert (
            type(confo_model) is pitci_xgb.XGBoosterAbsoluteErrorConformalPredictor
        ), "incorrect type returned from return_xgb_booster_absolute_error_confromal_predictor"

        assert confo_model.model is xgboost_1_split_1_tree, (
            "passed model arg not set to model attribute of object returned "
            "from return_xgb_booster_absolute_error_confromal_predictor"
        )

    def test_return_xgb_sklearn_absolute_error_confromal_predictor(
        self, xgb_regressor_1_split_1_tree
    ):
        """Test return_xgb_sklearn_absolute_error_confromal_predictor function returns
        an XGBSklearnAbsoluteErrorConformalPredictor object with model passed.
        """

        confo_model = pitci_xgb.return_xgb_sklearn_absolute_error_confromal_predictor(
            xgb_regressor_1_split_1_tree
        )

        assert (
            type(confo_model) is pitci_xgb.XGBSklearnAbsoluteErrorConformalPredictor
        ), "incorrect type returned from return_xgb_sklearn_absolute_error_confromal_predictor"

        assert confo_model.model is xgb_regressor_1_split_1_tree, (
            "passed model arg not set to model attribute of object returned "
            "from return_xgb_sklearn_absolute_error_confromal_predictor"
        )

    def test_return_xgb_booster_leaf_node_scaled_confromal_predictor(
        self, xgboost_1_split_1_tree
    ):
        """Test return_xgb_booster_leaf_node_scaled_confromal_predictor function returns
        an XGBoosterLeafNodeScaledConformalPredictor object with model passed.
        """

        confo_model = pitci_xgb.return_xgb_booster_leaf_node_scaled_confromal_predictor(
            xgboost_1_split_1_tree
        )

        assert (
            type(confo_model) is pitci_xgb.XGBoosterLeafNodeScaledConformalPredictor
        ), "incorrect type returned from return_xgb_booster_leaf_node_scaled_confromal_predictor"

        assert confo_model.model is xgboost_1_split_1_tree, (
            "passed model arg not set to model attribute of object returned "
            "from return_xgb_booster_leaf_node_scaled_confromal_predictor"
        )

    def test_return_xgb_sklearn_leaf_node_scaled_confromal_predictor(
        self, xgb_regressor_1_split_1_tree
    ):
        """Test return_xgb_sklearn_leaf_node_scaled_confromal_predictor function returns
        an XGBSklearnLeafNodeScaledConformalPredictor object with model passed.
        """

        confo_model = pitci_xgb.return_xgb_sklearn_leaf_node_scaled_confromal_predictor(
            xgb_regressor_1_split_1_tree
        )

        assert (
            type(confo_model) is pitci_xgb.XGBSklearnLeafNodeScaledConformalPredictor
        ), "incorrect type returned from return_xgb_sklearn_leaf_node_scaled_confromal_predictor"

        assert confo_model.model is xgb_regressor_1_split_1_tree, (
            "passed model arg not set to model attribute of object returned "
            "from return_xgb_sklearn_leaf_node_scaled_confromal_predictor"
        )
