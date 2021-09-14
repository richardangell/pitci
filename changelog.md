# Change Log

----

This changelog follows the great advice from https://keepachangelog.com/.

Each section will have a title of the format `X.Y.Z (YYYY-MM-DD) [#N #M]` giving the version of the package, the date of release of that version and a link to the relevant pull requests. Unreleased changes i.e. those that have been merged into master (e.g. with a .dev suffix) but which are not yet in a new release (on PyPI) are added to the changelog but with the title `X.Y.Z (unreleased)`. Unreleased sections can be combined when they are released and the date of release added to the title.

Subsections for each version can be one of the following;

- `Added` for new features.
- `Changed` for changes in existing functionality.
- `Deprecated` for soon-to-be removed features.
- `Removed` for now removed features.
- `Fixed` for any bug fixes.
- `Security` in case of vulnerabilities.

# 0.3.0.dev1 (unreleased)

## Changed

- Update changelog to follow structure recommendations from https://keepachangelog.com/
- Change file type of chaneglog to `.rst`
- Add changelog into sphinx docs

# 0.3.0.dev0 (unreleased) [#9](https://github.com/richardangell/pitci/pull/9)

## Added

- Add new `ConformalPredictor` abstract base class that all other conformal predictor classes will inherit from
  - Add `_lookup_baseline_interval` method in `ConformalPredictor` which returns the `baseline_interval` attribute but which can be overridden by the split conformal predictor classes or future classes where the baseline interval is not a constant value
- Add new tests; `TestConformalPredictionValues` for the model type specific `ConformalPredictor` subclasses that test (when using a non-trivial model);
  - The conformal predictor is calibrated at the expected level for different values of alpha
  - The conformal predictor gives the expected intervals
- Add new `docstrings.combine_split_mixin_docs` function to combine docstring for `SplitConformalPredictorMixin` and the model specific classes it will be jointly inherited with

## Changed

- Update `AbsoluteErrorConformalPredictor` and `LeafNodeScaledConformalPredictor` classes to inherit from `ConformalPredictor`
- Remove `predict_with_interval` and `_calibrate_interval` methods from `LeafNodeScaledConformalPredictor` class, these are now in the `ConformalPredictor` class
- Refactor `SplitConformalPredictor` into `SplitConformalPredictorMixin` that does not inherit from `LeafNodeScaledConformalPredictor`
  - Rename `baseline_intervals` attribute to `baseline_interval`
  - Remove `predict_with_interval` method
  - Remove `calibrate` method
- Revert the `nonconformity.nonconformity_at_alpha` function to use `np.quantile` but with `interpolation="higher"` to select the upper value if the quantile falls between two values
- Abstract out calculation of nonconformity scores into a `_calculate_nonconformity_scores` method which is implemented in `AbsoluteErrorConformalPredictor` and `LeafNodeScaledConformalPredictor` classes
- Change `_sum_dict_values` to be a staticmethod of `LeafNodeScaledConformalPredictor` rather than a function in `pitci.base`
- Change linting, tests and mypy to always run in the github actions pipeline
- Renamed `LGBMBoosterLeafNodeSplitConformalPredictor` to `LGBMBoosterSplitLeafNodeScaledConformalPredictor`
- Renamed `XGBoosterLeafNodeSplitConformalPredictor` to `XGBoosterSplitLeafNodeScaledConformalPredictor`
- Renamed `get_leaf_node_split_conformal_predictor` to `get_split_leaf_node_scaled_conformal_predictor`

# 0.2.0 (2021-07-26) [[#3](https://github.com/richardangell/pitci/pull/3) [#4](https://github.com/richardangell/pitci/pull/4) [#5](https://github.com/richardangell/pitci/pull/5) [#6](https://github.com/richardangell/pitci/pull/6) [#7](https://github.com/richardangell/pitci/pull/7)]

## Added

- Add `train_data` argument to the `calibrate` methods of `XGBoosterLeafNodeScaledConformalPredictor` and `XGBSklearnLeafNodeScaledConformalPredictor` classes to allow the user to calibrate the leaf node counts on a different (train) data sample, rather than the sample used to calibrate the interval widths (which shouldn't be the training sample) [#3](https://github.com/richardangell/pitci/pull/3)
- Add `LGBMBoosterLeafNodeScaledConformalPredictor` class to provide leaf node count scaled conformal intervals for `lgb.Booster` models [#4](https://github.com/richardangell/pitci/pull/4)
- Add `sphinx` documentation for package in `docs` folder [#5](https://github.com/richardangell/pitci/pull/5)
- Add `SplitConformalPredictor` class that allows conformal intervals to be calibrated for different bands of the data based off the scaling factor [#6](https://github.com/richardangell/pitci/pull/6)
- Add `XGBoosterLeafNodeSplitConformalPredictor` class that allows split conformal intervals with `xgb.Booster` objects where the scaling factor is based off the leaf node counts [#6](https://github.com/richardangell/pitci/pull/6)
- Add `LGBMBoosterLeafNodeSplitConformalPredictor` class that allows split conformal intervals with `lgb.Booster` objects where the scaling factor is based off the leaf node counts [#6](https://github.com/richardangell/pitci/pull/6)
- Consolidate docstrings across inherited classes with new `docstrings` module [#7](https://github.com/richardangell/pitci/pull/7)

## Changed

- Remove `xgboost` and add `pandas` to `requirements.txt` [#4](https://github.com/richardangell/pitci/pull/4)
- Swap project to use `flit` as the package build tool [#7](https://github.com/richardangell/pitci/pull/7)
- Change calculation of `alpha` at given quantile to select closest observation if the quantile falls between two values [#7](https://github.com/richardangell/pitci/pull/7)

# 0.1.1 (2021-05-06) [[#1](https://github.com/richardangell/pitci/pull/1)]

## Added

- Add support for `xgb.XGBRegressor` and `xgb.XGBClassifier` objects with non scaled nonconformity measure in `XGBSklearnAbsoluteErrorConformalPredictor` class
- Add support for `xgb.XGBRegressor` and `xgb.XGBClassifier` objects with leaf node scaled nonconformity measure in `XGBSklearnLeafNodeScaledConformalPredictor` class
- Add `dispatches` module with helper functions `get_absolute_error_conformal_predictor` and `get_leaf_node_scaled_conformal_predictor` to return correct conformal predictor class given the type of underlying model passed

## Changed

- Change `AbsoluteErrorConformalPredictor` to be abstract base class
- Add `XGBoosterAbsoluteErrorConformalPredictor` class to provide non scaled conformal intervals for `xgb.Booster` objects, previously the `AbsoluteErrorConformalPredictor` class provided this functionality
- Change `LeafNodeScaledConformalPredictor` to be abstract base class
- Add `XGBoosterLeafNodeScaledConformalPredictor` class to provide leaf node scaled conformal intervals for `xgb.Booster` objects, previously the `LeafNodeScaledConformalPredictor` class provided this functionality

# 0.1.0 (2021-05-01)

## Added

- Add `AbsoluteErrorConformalPredictor` class implementing non scaled conformal intervals for `xgb.Booster` objects 
- Add `LeafNodeScaledConformalPredictor` class implementing conformal intervals scaled by leaf node counts for `xgb.Booster` objects
