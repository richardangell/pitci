"""Prediction intervals for trees using conformal intervals - pitci"""

from ._version import __version__

from . import base
from . import checks
from . import dispatchers
from . import helpers

from .dispatchers import (
    get_leaf_node_scaled_conformal_predictor,
    get_absolute_error_conformal_predictor,
    get_leaf_node_split_conformal_predictor,
)

try:
    from . import xgboost
except ImportError:
    pass

try:
    from . import lightgbm
except ImportError:
    pass
