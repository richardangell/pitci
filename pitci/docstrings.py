"""
Module providing funtionality for docstring inheriting solution used in the package.

Note, other methods were consider to achieve the goal of the `doc_inherit_kwargs` i.e.
allow child class methods to inherit the docstring from the parent class method but
apply string formatting to the child method so that the parent docstring acts as a
template.  One example was;
https://code.activestate.com/recipes/578587-inherit-method-docstrings-without-breaking-decorat/,
however adapting `custom_inherit` proved to be more simple.

"""

from . import base

from functools import partial
import custom_inherit

from typing import Any, Union, Callable


def _format_base_class_docstrings() -> None:
    """Function to format the docstrings for the base conformal
    predictor classes in the base module.

    This stops the string formatting keys showing up in the
    documentation.

    """

    _format_absolute_error_conformal_predictor()

    _format_leaf_node_scaled_absolute_error_conformal_predictor()

    _format_leaf_node_split_absolute_error_conformal_predictor()


def _format_absolute_error_conformal_predictor() -> None:
    """Format the class docstring and user facing methods docstrings
    in the AbsoluteErrorConformalPredictor class.
    """

    _str_format_docstring(
        base.AbsoluteErrorConformalPredictor,
        model_type="``Any``",
        description="",
        parameters="",
        calibrate_link=":func:`~pitci.base.AbsoluteErrorConformalPredictor.calibrate`",
        attributes="",
    )

    _str_format_docstring(
        base.AbsoluteErrorConformalPredictor.calibrate,
        description="",
        data_type="``Any``",
        response_type="np.ndarray or pd.Series",
    )

    _str_format_docstring(
        base.AbsoluteErrorConformalPredictor.predict_with_interval,
        description="",
        data_type="``Any``",
    )


def _format_leaf_node_scaled_absolute_error_conformal_predictor() -> None:
    """Format the class docstring and user facing methods docstrings
    in the LeafNodeScaledConformalPredictor class.
    """

    _str_format_docstring(
        base.LeafNodeScaledConformalPredictor,
        model_type="``Any``",
        description="",
        parameters="",
        attributes="",
    )

    _str_format_docstring(
        base.LeafNodeScaledConformalPredictor.calibrate,
        predict_with_interval_method=":func:`~pitci.base.LeafNodeScaledConformalPredictor.predict_with_interval`",
        data_type="``Any``",
        response_type="np.ndarray or pd.Series",
        train_data_type="Any, default = None",
        description="",
    )

    _str_format_docstring(
        base.LeafNodeScaledConformalPredictor.predict_with_interval,
        data_type="``Any``",
    )


def _format_leaf_node_split_absolute_error_conformal_predictor() -> None:
    """Format the class docstring and user facing methods docstrings
    in the SplitConformalPredictor class.
    """

    _str_format_docstring(
        base.SplitConformalPredictor,
        model_type="``Any``",
        description="",
        parameters="",
        calibrate_link="``calibrate``",
        attributes="",
    )


def _str_format_docstring(obj: Any, **kwargs) -> None:
    """Format a class or methods docstring using the str.format method
    with the keyword arguments passed.

    obj : Any
        An object with a __doc__ attribute to apply str.format to.

    """

    obj.__doc__ = obj.__doc__.format(**kwargs)


def doc_inherit_kwargs(
    parent: Union[str, Any],
    style: Union[Any, Callable[[str, str], str]] = "parent",
    **kwargs
) -> custom_inherit._decorator_base.DocInheritDecorator:
    """This function is a slight modification of the `doc_inherit` decorator
    function from the `custom_inherit` package to allow keyword arguments to
    be passed into the function.

    Having keyword arguments passed into the `merge_func` allows the function
    to apply the `str.format` method to allow child method docstrings to be
    created based off of i.e. formatted from the template docstring in the
    parent class method. See the `str_format_merge_style` example below.

    Original documentation from `custom_inherit.doc_inherit` below;

    Returns a function/method decorator that, given `parent`, updates the docstring
    of the decorate function/method based on the specified style and the corresponding
    attribute of `parent`.

    Parameters
    ----------
    parent : Union[str, Any]
        The docstring, or object of which the docstring is utilized as the
        parent docstring during the docstring merge.

    style : Union[Any, Callable[[str, str], str]], optional (default: "parent")
        A valid inheritance-scheme style ID or function that merges two docstrings.

    Returns
    -------
    custom_inherit.DocInheritDecorator

    Notes
    -----
    `doc_inherit` should always be used as the inner-most decorator when being used in
    conjunction with other decorators, such as `@property`, `@staticmethod`, etc.
    """

    merge_func = custom_inherit.store[style]
    decorator = custom_inherit._DocInheritDecorator
    decorator.doc_merger = staticmethod(partial(merge_func, **kwargs))
    return decorator(parent)


def str_format_merge_style(prnt_doc: str, child_doc: str, **kwargs) -> str:
    """Custom docstring merge function that applies `str.format` to the parent
    class docstring (`prnt_doc`) with the keyword arguments passed to the
    function.
    """

    return prnt_doc.format(**kwargs)
