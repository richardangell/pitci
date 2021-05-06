# Change Log
----

# 0.1.1

- Change `AbsoluteErrorConformalPredictor` to be abstract base class
- Add `XGBoosterAbsoluteErrorConformalPredictor` class to provide leaf node scaled conformal intervals for `xgb.Booster` objects  
- Add support for `xgb.XGBRegressor` and `xgb.XGBClassifier` objects with non scaled nonconformity measure in `XGBSklearnAbsoluteErrorConformalPredictor` class
- Change `LeafNodeScaledConformalPredictor` to be abstract base class
- Add `XGBoosterLeafNodeScaledConformalPredictor` class to provide leaf node scaled conformal intervals for `xgb.Booster` objects  
- Add support for `xgb.XGBRegressor` and `xgb.XGBClassifier` objects with leaf node scaled nonconformity measure in `XGBSklearnLeafNodeScaledConformalPredictor` class
- Add `dispatches` module with helper functions `get_absolute_error_conformal_predictor` and `get_leaf_node_scaled_conformal_predictor` to return correct class given the type of underlying model passed

# 0.1.0

- Add `AbsoluteErrorConformalPredictor` class implementing non scaled conformal intervals for `xgb.Booster` objects 
- Add `LeafNodeScaledConformalPredictor` class implementing conformal intervals scaled by leaf node counts for `xgb.Booster` objects


