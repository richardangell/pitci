import re
import xgboost as xgb

import pitci.checks as checks

import pytest


class TestCheckType:
    """Tests for the check_type function."""

    @pytest.mark.parametrize(
        "obj, expected_types, obj_name, exception_text",
        [
            (
                1,
                [float],
                "sss",
                "sss is not in expected types [<class 'float'>], got <class 'int'>",
            ),
            (
                1.0,
                [int],
                "abc",
                "abc is not in expected types [<class 'int'>], got <class 'float'>",
            ),
            (
                "sdkksd",
                [int, float],
                "name",
                "name is not in expected types [<class 'int'>, <class 'float'>], got <class 'str'>",
            ),
        ],
    )
    def test_type_exception_raised(self, obj, expected_types, obj_name, exception_text):
        """Test an exception is raised if obj is not of the correct type(s)."""

        with pytest.raises(TypeError, match=re.escape(exception_text)):

            checks.check_type(obj, expected_types, obj_name)

    def test_non_list_exception(self):
        """Test an exception is raised if expected_types is not a list."""

        with pytest.raises(TypeError, match=re.escape("expected_types must be a list")):

            checks.check_type(1, 1, "1")

    def test_non_type_exception(self):
        """Test an exception is raised if not all of expected_types elements are types."""

        with pytest.raises(
            TypeError, match=re.escape("all elements in expected_types must be types")
        ):

            checks.check_type(1, [int, 1], "1")


class TestCheckAttribute:
    """Tests for the check_attribute function."""

    @pytest.mark.parametrize(
        "obj, attribute, message", [([], "keys", "ssssss"), ({}, "index", "abc")]
    )
    def test_exception_raised(self, obj, attribute, message):
        """Test an exception is raised if obj does not have the given attribute."""

        with pytest.raises(AttributeError, match=re.escape(message)):

            checks.check_attribute(obj, attribute, message)


class TestCheckAllowedValue:
    """Tests for the check_allowed_value function."""

    @pytest.mark.parametrize(
        "value, allowed_values, message",
        [("x", [], "ssssss"), (1, [2, "b", False], "abc")],
    )
    def test_exception_raised(self, value, allowed_values, message):
        """Test an exception is raised if value is not in allowed_values."""

        error_message = f"{message}\n{value} not in allowed values; {allowed_values}"

        with pytest.raises(ValueError, match=re.escape(error_message)):

            checks.check_allowed_value(value, allowed_values, message)


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

            checks.check_objective_supported(model, supported_objectives)
