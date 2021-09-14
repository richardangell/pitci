Changelog
=========

This changelog follows the great advice from https://keepachangelog.com/.

Each section will have a title of the format ``X.Y.Z (YYYY-MM-DD)`` giving the version of the package and the date of release of that version. Unreleased changes i.e. those that have been merged into master (e.g. with a .dev suffix) but which are not yet in a new release (on PyPI) are added to the changelog but with the title ``X.Y.Z (unreleased)``. Unreleased sections can be combined when they are released and the date of release added to the title.

Subsections for each version can be one of the following;

- ``Added`` for new features.
- ``Changed`` for changes in existing functionality.
- ``Deprecated`` for soon-to-be removed features.
- ``Removed`` for now removed features.
- ``Fixed`` for any bug fixes.
- ``Security`` in case of vulnerabilities.

Each individual change should have a link to the pull request after the description of the change.

0.3.0.dev1 (unreleased)
-----------------------

Changed
^^^^^^^

- Update changelog to follow structure recommendations from https://keepachangelog.com/
- Change file type of chaneglog to ``.rst``

Added
^^^^^

- Add changelog into sphinx docs

0.3.0.dev0 (unreleased) 
-----------------------

Added
^^^^^

- Add new ``ConformalPredictor`` abstract base class that all other conformal predictor classes will inherit from `#9 <https://github.com/richardangell/pitci/pull/9>`_
    - Add `_lookup_baseline_interval`` method in ``ConformalPredictor`` which returns the ``baseline_interval`` attribute but which can be overridden by the split conformal predictor classes or future classes where the baseline interval is not a constant value
- Add new tests; ``TestConformalPredictionValues`` for the model type specific ``ConformalPredictor`` subclasses that test (when using a non-trivial model) `#9 <https://github.com/richardangell/pitci/pull/9>`_
    - The conformal predictor is calibrated at the expected level for different values of alpha
    - The conformal predictor gives the expected intervals
- Add new ``docstrings.combine_split_mixin_docs`` function to combine docstring for ``SplitConformalPredictorMixin`` and the model specific classes it will be jointly inherited with `#9 <https://github.com/richardangell/pitci/pull/9>`_

Changed
^^^^^^^

- Update ``AbsoluteErrorConformalPredictor`` and ``LeafNodeScaledConformalPredictor`` classes to inherit from ``ConformalPredictor`` `#9 <https://github.com/richardangell/pitci/pull/9>`_
- Remove ``predict_with_interval`` and `_calibrate_interval`` methods from ``LeafNodeScaledConformalPredictor`` class, these are now in the ``ConformalPredictor`` class `#9 <https://github.com/richardangell/pitci/pull/9>`_
- Refactor ``SplitConformalPredictor`` into ``SplitConformalPredictorMixin`` that does not inherit from ``LeafNodeScaledConformalPredictor`` `#9 <https://github.com/richardangell/pitci/pull/9>`_
    - Rename ``baseline_intervals`` attribute to ``baseline_interval``
    - Remove ``predict_with_interval`` method
    - Remove ``calibrate`` method
- Revert the ``nonconformity.nonconformity_at_alpha`` function to use ``np.quantile`` but with ``interpolation="higher"`` to select the upper value if the quantile falls between two values `#9 <https://github.com/richardangell/pitci/pull/9>`_
- Abstract out calculation of nonconformity scores into a `_calculate_nonconformity_scores`` method which is implemented in ``AbsoluteErrorConformalPredictor`` and ``LeafNodeScaledConformalPredictor`` classes `#9 <https://github.com/richardangell/pitci/pull/9>`_
- Change ``_sum_dict_values`` to be a staticmethod of ``LeafNodeScaledConformalPredictor`` rather than a function in ``pitci.base`` `#9 <https://github.com/richardangell/pitci/pull/9>`_
- Change linting, tests and mypy to always run in the github actions pipeline `#9 <https://github.com/richardangell/pitci/pull/9>`_
- Renamed ``LGBMBoosterLeafNodeSplitConformalPredictor`` to ``LGBMBoosterSplitLeafNodeScaledConformalPredictor`` `#9 <https://github.com/richardangell/pitci/pull/9>`_
- Renamed ``XGBoosterLeafNodeSplitConformalPredictor`` to ``XGBoosterSplitLeafNodeScaledConformalPredictor`` `#9 <https://github.com/richardangell/pitci/pull/9>`_
- Renamed ``get_leaf_node_split_conformal_predictor`` to ``get_split_leaf_node_scaled_conformal_predictor`` `#9 <https://github.com/richardangell/pitci/pull/9>`_

0.2.0 (2021-07-26)
------------------

Added
^^^^^

- Add ``train_data`` argument to the ``calibrate`` methods of ``XGBoosterLeafNodeScaledConformalPredictor`` and ``XGBSklearnLeafNodeScaledConformalPredictor`` classes to allow the user to calibrate the leaf node counts on a different (train) data sample, rather than the sample used to calibrate the interval widths (which shouldn't be the training sample) `#3 <https://github.com/richardangell/pitci/pull/3>`_
- Add ``LGBMBoosterLeafNodeScaledConformalPredictor`` class to provide leaf node count scaled conformal intervals for ``lgb.Booster`` models `#4 <https://github.com/richardangell/pitci/pull/4>`_
- Add ``sphinx`` documentation for package in ``docs`` folder `#5 <https://github.com/richardangell/pitci/pull/5>`_
- Add ``SplitConformalPredictor`` class that allows conformal intervals to be calibrated for different bands of the data based off the scaling factor `#6 <https://github.com/richardangell/pitci/pull/6>`_
- Add ``XGBoosterLeafNodeSplitConformalPredictor`` class that allows split conformal intervals with ``xgb.Booster`` objects where the scaling factor is based off the leaf node counts `#6 <https://github.com/richardangell/pitci/pull/6>`_
- Add ``LGBMBoosterLeafNodeSplitConformalPredictor`` class that allows split conformal intervals with ``lgb.Booster`` objects where the scaling factor is based off the leaf node counts `#6 <https://github.com/richardangell/pitci/pull/6>`_
- Consolidate docstrings across inherited classes with new ``docstrings`` module `#7 <https://github.com/richardangell/pitci/pull/7>`_

Changed
^^^^^^^

- Remove ``xgboost`` and add ``pandas`` to ``requirements.txt`` `#4 <https://github.com/richardangell/pitci/pull/4>`_
- Swap project to use ``flit`` as the package build tool `#7 <https://github.com/richardangell/pitci/pull/7>`_
- Change calculation of ``alpha`` at given quantile to select closest observation if the quantile falls between two values `#7 <https://github.com/richardangell/pitci/pull/7>`_

0.1.1 (2021-05-06)
------------------

Added
^^^^^

- Add support for ``xgb.XGBRegressor`` and ``xgb.XGBClassifier`` objects with non scaled nonconformity measure in ``XGBSklearnAbsoluteErrorConformalPredictor`` class `#1 <https://github.com/richardangell/pitci/pull/1>`_
- Add support for ``xgb.XGBRegressor`` and ``xgb.XGBClassifier`` objects with leaf node scaled nonconformity measure in ``XGBSklearnLeafNodeScaledConformalPredictor`` class `#1 <https://github.com/richardangell/pitci/pull/1>`_
- Add ``dispatches`` module with helper functions ``get_absolute_error_conformal_predictor`` and ``get_leaf_node_scaled_conformal_predictor`` to return correct conformal predictor class given the type of underlying model passed `#1 <https://github.com/richardangell/pitci/pull/1>`_

Changed
^^^^^^^

- Change ``AbsoluteErrorConformalPredictor`` to be abstract base class `#1 <https://github.com/richardangell/pitci/pull/1>`_
- Add ``XGBoosterAbsoluteErrorConformalPredictor`` class to provide non scaled conformal intervals for ``xgb.Booster`` objects, previously the ``AbsoluteErrorConformalPredictor`` class provided this functionality `#1 <https://github.com/richardangell/pitci/pull/1>`_
- Change ``LeafNodeScaledConformalPredictor`` to be abstract base class `#1 <https://github.com/richardangell/pitci/pull/1>`_
- Add ``XGBoosterLeafNodeScaledConformalPredictor`` class to provide leaf node scaled conformal intervals for ``xgb.Booster`` objects, previously the ``LeafNodeScaledConformalPredictor`` class provided this functionality `#1 <https://github.com/richardangell/pitci/pull/1>`_

0.1.0 (2021-05-01)
------------------

Added
^^^^^

- Add ``AbsoluteErrorConformalPredictor`` class implementing non scaled conformal intervals for ``xgb.Booster`` objects 
- Add ``LeafNodeScaledConformalPredictor`` class implementing conformal intervals scaled by leaf node counts for ``xgb.Booster`` objects
