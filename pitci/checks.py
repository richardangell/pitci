import xgboost as xgb
import json

from typing import Any, Type, List


def check_type(obj: Any, expected_types: List[Type], obj_name: str) -> None:
    """Function to check object is of given types and raise a TypeError
    if not.
    """

    if type(expected_types) is not list:

        raise TypeError("expected_types must be a list")

    if not all([type(expected_type) is type for expected_type in expected_types]):

        raise TypeError("all elements in expected_types must be types")

    if type(obj) not in expected_types:

        raise TypeError(
            f"{obj_name} is not in expected types {expected_types}, got {type(obj)}"
        )


def check_attribute(obj: Any, attribute: str, message: str) -> None:
    """Function to check an object has a given attribute and raise an
    AttributeError with specific message if not.
    """

    check_type(attribute, [str], "attribute")
    check_type(message, [str], "message")

    if not hasattr(obj, attribute):

        raise AttributeError(message)


def check_allowed_value(
    value: Any, allowed_values: List[Any], message: str = ""
) -> None:
    """Function to check that a value is in a list of allowed values
    and raise an exception if not.
    """

    check_type(allowed_values, [list], "allowed_values")
    check_type(message, [str], "message")

    if value not in allowed_values:

        raise ValueError(f"{message}\n{value} not in allowed values; {allowed_values}")


def check_objective_supported(
    booster: xgb.Booster, supported_objectives: List[str]
) -> None:
    """Function to check that the booster objective parameter is in the
    supported_objectives list and raise and exception if not.
    """

    check_type(booster, [xgb.Booster], "booster")
    check_type(supported_objectives, [list], "supported_objectives")

    for i, objective in enumerate(supported_objectives):

        check_type(objective, [str], f"supported_objectives[{i}]")

    booster_config = json.loads(booster.save_config())

    booster_objective = booster_config["learner"]["objective"]["name"]

    check_allowed_value(
        booster_objective, supported_objectives, "booster objective not supported"
    )
