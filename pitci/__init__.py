"""Prediction intervals for trees using conformal intervals - pitci"""

from ._version import __version__

from . import base
from . import checks
from . import dispatchers
from . import helpers

from .dispatchers import (
    get_leaf_node_scaled_conformal_predictor,
    get_absolute_error_conformal_predictor,
    get_split_leaf_node_scaled_conformal_predictor,
)

from .docstrings import _format_base_class_docstrings

try:
    from . import xgboost
except ImportError:
    pass

try:
    from . import lightgbm
except ImportError:
    pass

# format docs for base conformal predictors after all the other
# modules have been imported and they have formatted the base
# docstrings for their classes
_format_base_class_docstrings()
