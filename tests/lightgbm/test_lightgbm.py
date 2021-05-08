import lightgbm as lgb
import re
import pytest

import pitci.lightgbm as pitci_lgb


class TestCheckObjectiveSupported:
    """Tests for the check_objective_supported function."""

    @pytest.mark.parametrize(
        "objective, supported_objectives, message",
        [
            ("regression", ["huber", "fair"], "test"),
            ("regression_l1", ["poisson", "quantile"], "xyz"),
        ],
    )
    def test_exception_raised(
        self, lgb_dataset_2x1_with_label, objective, supported_objectives, message
    ):
        """Test an exception is raised if a model with an object not in the
        supported_objective list.
        """

        params = {
            "objective": objective,
            "num_leaves": 2,
            "min_data_in_leaf": 1,
            "feature_pre_filter": False,
        }

        model = lgb.train(
            params=params, train_set=lgb_dataset_2x1_with_label, num_boost_round=1
        )

        error_message = f"booster objective not supported\n{objective} not in allowed values; {supported_objectives}"

        with pytest.raises(
            ValueError,
            match=re.escape(error_message),
        ):

            pitci_lgb.check_objective_supported(model, supported_objectives)
