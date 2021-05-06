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
