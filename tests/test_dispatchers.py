import re
import pytest

import pitci.xgboost as pitci_xgb
import pitci.lightgbm as pitci_lgb
import pitci.dispatchers as dispatchers


class TestGetAbsoluteErrorConformalPredictor:
    """Tests for the get_absolute_error_conformal_predictor function."""

    def test_xgb_booster(self, xgboost_1_split_1_tree):
        """Test an XGBoosterAbsoluteErrorConformalPredictor object is returned if
        and xgb.Booster is passed.
        """

        confo_model = dispatchers.get_absolute_error_conformal_predictor(
            xgboost_1_split_1_tree
        )

        assert (
            type(confo_model) is pitci_xgb.XGBoosterAbsoluteErrorConformalPredictor
        ), "incorrect type returned from get_absolute_error_conformal_predictor"

        assert confo_model.model is xgboost_1_split_1_tree, (
            "passed model arg not set to model attribute of object returned "
            "from get_absolute_error_conformal_predictor"
        )

    def test_xgb_regressor(self, xgb_regressor_1_split_1_tree):
        """Test an XGBSklearnAbsoluteErrorConformalPredictor object is returned if
        and xgb.XGBRegressor is passed.
        """

        confo_model = dispatchers.get_absolute_error_conformal_predictor(
            xgb_regressor_1_split_1_tree
        )

        assert (
            type(confo_model) is pitci_xgb.XGBSklearnAbsoluteErrorConformalPredictor
        ), "incorrect type returned from get_absolute_error_conformal_predictor"

        assert confo_model.model is xgb_regressor_1_split_1_tree, (
            "passed model arg not set to model attribute of object returned "
            "from get_absolute_error_conformal_predictor"
        )

    def test_xgb_classifier(self, xgb_classifier_1_split_1_tree):
        """Test an XGBSklearnAbsoluteErrorConformalPredictor object is returned if
        and xgb.XGBClassifier is passed.
        """

        confo_model = dispatchers.get_absolute_error_conformal_predictor(
            xgb_classifier_1_split_1_tree
        )

        assert (
            type(confo_model) is pitci_xgb.XGBSklearnAbsoluteErrorConformalPredictor
        ), "incorrect type returned from get_absolute_error_conformal_predictor"

        assert confo_model.model is xgb_classifier_1_split_1_tree, (
            "passed model arg not set to model attribute of object returned "
            "from get_absolute_error_conformal_predictor"
        )

    def test_other_type_exception(self):
        """Test an exception is raised if a non-implemented type is passed."""

        with pytest.raises(
            NotImplementedError,
            match=re.escape(
                f"model type not supported for AbsoluteErrorConformalPredictor children; {int}"
            ),
        ):

            dispatchers.get_absolute_error_conformal_predictor(12345)


class TestGetLeafNodeScaledConformalPredictor:
    """Tests for the get_leaf_node_scaled_conformal_predictor function."""

    def test_xgb_booster(self, xgboost_1_split_1_tree):
        """Test an XGBoosterLeafNodeScaledConformalPredictor object is returned if
        and xgb.Booster is passed.
        """

        confo_model = dispatchers.get_leaf_node_scaled_conformal_predictor(
            xgboost_1_split_1_tree
        )

        assert (
            type(confo_model) is pitci_xgb.XGBoosterLeafNodeScaledConformalPredictor
        ), "incorrect type returned from get_leaf_node_scaled_conformal_predictor"

        assert confo_model.model is xgboost_1_split_1_tree, (
            "passed model arg not set to model attribute of object returned "
            "from get_leaf_node_scaled_conformal_predictor"
        )

    def test_xgb_regressor(self, xgb_regressor_1_split_1_tree):
        """Test an XGBSklearnLeafNodeScaledConformalPredictor object is returned if
        and xgb.XGBRegressor is passed.
        """

        confo_model = dispatchers.get_leaf_node_scaled_conformal_predictor(
            xgb_regressor_1_split_1_tree
        )

        assert (
            type(confo_model) is pitci_xgb.XGBSklearnLeafNodeScaledConformalPredictor
        ), "incorrect type returned from get_leaf_node_scaled_conformal_predictor"

        assert confo_model.model is xgb_regressor_1_split_1_tree, (
            "passed model arg not set to model attribute of object returned "
            "from get_leaf_node_scaled_conformal_predictor"
        )

    def test_xgb_classifier(self, xgb_classifier_1_split_1_tree):
        """Test an XGBSklearnLeafNodeScaledConformalPredictor object is returned if
        and xgb.XGBClassifier is passed.
        """

        confo_model = dispatchers.get_leaf_node_scaled_conformal_predictor(
            xgb_classifier_1_split_1_tree
        )

        assert (
            type(confo_model) is pitci_xgb.XGBSklearnLeafNodeScaledConformalPredictor
        ), "incorrect type returned from get_leaf_node_scaled_conformal_predictor"

        assert confo_model.model is xgb_classifier_1_split_1_tree, (
            "passed model arg not set to model attribute of object returned "
            "from get_leaf_node_scaled_conformal_predictor"
        )

    def test_lgb_booster(self, lgb_booster_1_split_1_tree):
        """Test an LGBMBoosterLeafNodeScaledConformalPredictor object is returned if
        and lgb.Booster is passed.
        """

        confo_model = dispatchers.get_leaf_node_scaled_conformal_predictor(
            lgb_booster_1_split_1_tree
        )

        assert (
            type(confo_model) is pitci_lgb.LGBMBoosterLeafNodeScaledConformalPredictor
        ), "incorrect type returned from get_leaf_node_scaled_conformal_predictor"

        assert confo_model.model is lgb_booster_1_split_1_tree, (
            "passed model arg not set to model attribute of object returned "
            "from get_leaf_node_scaled_conformal_predictor"
        )

    def test_other_type_exception(self):
        """Test an exception is raised if a non-implemented type is passed."""

        with pytest.raises(
            NotImplementedError,
            match=re.escape(
                f"model type not supported for LeafNodeScaledConformalPredictor children; {float}"
            ),
        ):

            dispatchers.get_leaf_node_scaled_conformal_predictor(1.1)


class TestGetLeafNodeSplitConformalPredictor:
    """Tests for the get_leaf_node_split_conformal_predictor function."""

    def test_xgb_booster(self, xgboost_1_split_1_tree):
        """Test an XGBoosterSplitLeafNodeScaledConformalPredictor object is returned if
        and xgb.Booster is passed.
        """

        confo_model = dispatchers.get_leaf_node_split_conformal_predictor(
            xgboost_1_split_1_tree, 5
        )

        assert (
            type(confo_model)
            is pitci_xgb.XGBoosterSplitLeafNodeScaledConformalPredictor
        ), "incorrect type returned from get_leaf_node_split_conformal_predictor"

        assert confo_model.model is xgboost_1_split_1_tree, (
            "passed model arg not set to model attribute of object returned "
            "from get_leaf_node_scaled_conformal_predictor"
        )

        assert confo_model.n_bins == 5, (
            "passed n_bins arg not set to n_bins attribute of object returned "
            "from get_leaf_node_split_conformal_predictor"
        )

    def test_lgb_booster(self, lgb_booster_1_split_1_tree):
        """Test an LGBMBoosterSplitLeafNodeScaledConformalPredictor object is returned if
        and lgb.Booster is passed.
        """

        confo_model = dispatchers.get_leaf_node_split_conformal_predictor(
            lgb_booster_1_split_1_tree, 8
        )

        assert (
            type(confo_model)
            is pitci_lgb.LGBMBoosterSplitLeafNodeScaledConformalPredictor
        ), "incorrect type returned from get_leaf_node_split_conformal_predictor"

        assert confo_model.model is lgb_booster_1_split_1_tree, (
            "passed model arg not set to model attribute of object returned "
            "from get_leaf_node_split_conformal_predictor"
        )

        assert confo_model.n_bins == 8, (
            "passed n_bins arg not set to n_bins attribute of object returned "
            "from get_leaf_node_split_conformal_predictor"
        )

    def test_other_type_exception(self):
        """Test an exception is raised if a non-implemented type is passed."""

        with pytest.raises(
            NotImplementedError,
            match=re.escape(
                f"model type not supported for SplitConformalPredictor children; {float}"
            ),
        ):

            dispatchers.get_leaf_node_split_conformal_predictor(1.1)
