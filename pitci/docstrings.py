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
from custom_inherit._doc_parse_tools.numpy_parse_tools import parse_numpy_doc

from typing import Any, Union, Callable


def _format_base_class_docstrings() -> None:
    """Function to format the docstrings for the base conformal
    predictor classes in the base module.

    This stops the string formatting keys showing up in the
    documentation.

    """

    _format_absolute_error_conformal_predictor()

    _format_leaf_node_scaled_absolute_error_conformal_predictor()


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
        calibrate_method="pitci.base.LeafNodeScaledConformalPredictor.calibrate",
    )

    _str_format_docstring(
        base.LeafNodeScaledConformalPredictor.calibrate,
        predict_with_interval_method="pitci.base.LeafNodeScaledConformalPredictor.predict_with_interval",
        data_type="``Any``",
        response_type="np.ndarray or pd.Series",
        train_data_type="Any, default = None",
        description="",
    )

    _str_format_docstring(
        base.LeafNodeScaledConformalPredictor.predict_with_interval,
        data_type="``Any``",
    )


def _str_format_docstring(obj: Any, **kwargs) -> None:
    """Format a class or methods docstring using the str.format method
    with the keyword arguments passed.

    obj : Any
        An object with a __doc__ attribute to apply str.format to.

    """

    obj.__doc__ = obj.__doc__.format(**kwargs)


def combine_split_mixin_docs(mixin_class, main_class):
    """Function to combine docstrings to produce a composite
    docstring for a model specific SplitLeafNodeScaledConformalPredictor
    child class.

    The specific classes this function should be used with are a
    model specific LeafNodeScaledConformalPredictor child class and
    the SplitConformalPredictorMixin class.

    """

    SPLIT_CONFORMAL_PREDICTOR_DESCRIPTION = (
        "Intervals are split into bins, using the scaling factors, where each bin is calibrated "
        "at the required confidence level. This addresses the situation where the leaf node "
        "scaled conformal predictors are not well calibrated on subsets of the data, despite "
        "being calibrated at the required ``alpha`` confidence level overall."
    )

    main_docs_split = parse_numpy_doc(main_class.__doc__)

    mixin_docs_split = parse_numpy_doc(mixin_class.__doc__)

    combined_docs = (
        main_docs_split["Short Summary"]
        + "\n\n"
        + SPLIT_CONFORMAL_PREDICTOR_DESCRIPTION
        + "\n\n"
        + "Parameters\n----------"
        + "\n\n"
        + main_docs_split["Parameters"]
        + "\n\n"
        + remove_key_from_numpy_docstring_section(
            mixin_docs_split["Parameters"], "model"
        )
        + "\n\n"
        + "Attributes\n----------"
        + "\n\n"
        + remove_key_from_numpy_docstring_section(
            main_docs_split["Attributes"], "baseline_interval"
        )
        + "\n\n"
        + mixin_docs_split["Attributes"]
    )

    return combined_docs


def remove_key_from_numpy_docstring_section(numpy_docstring_section, key):
    """Function to remove a specific key from a section of a numpy docstring.

    For example when combining docstrings with ``combine_split_mixin_docs``
    both docstrings may have a particular attribute listed so it is
    neccessary to remove one when merging the two.

    """

    docstring_section_split = numpy_docstring_section.split("\n")

    key_location = -1

    # first find the location of the list element containing "key :"
    for docstring_section_single_no, docstring_section_single in enumerate(
        docstring_section_split
    ):

        key_location_section_single = docstring_section_single.find(f"{key} :")

        if key_location_section_single >= 0:

            if key_location >= 0:

                raise ValueError(
                    f"key (specifically '{key } :') found twice in numpy_docstring_section"
                )

            else:

                key_location = docstring_section_single_no

    if key_location < 0:

        raise ValueError(
            f"key (specifically '{key } :') is not present in numpy_docstring_section"
        )

    delete_keys = []

    # next find the elements after the "key :" element which are the description
    # for the key
    # note, this can be multiple elements as the description can be split over
    # multiple lines
    # search from key_location until the next "" value or end of the list
    for docstring_section_single_no in range(
        key_location, len(docstring_section_split) - 1
    ):

        delete_keys.append(docstring_section_single_no)

        if docstring_section_split[docstring_section_single_no] == "":

            break

    # delete the key name and the key description lines
    for delete_key in reversed(delete_keys):

        del docstring_section_split[delete_key]

    modified_docstring_section = "\n".join(docstring_section_split)

    return modified_docstring_section


def doc_inherit_kwargs(
    parent: Union[str, Any],
    style: Union[Any, Callable[[str, str], str]] = "parent",
    **kwargs,
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
