api documentation
====================

.. currentmodule:: pitci

base module
------------------

.. autosummary::
    :toctree: api/

    base.AbsoluteErrorConformalPredictor
    base.LeafNodeScaledConformalPredictor
    base.SplitConformalPredictorMixin
    
lightgbm module
------------------

.. autosummary::
    :toctree: api/

    lightgbm.LGBMBoosterLeafNodeScaledConformalPredictor
    lightgbm.LGBMBoosterSplitLeafNodeScaledConformalPredictor
         
xgboost module
------------------

.. autosummary::
    :toctree: api/

    xgboost.XGBoosterAbsoluteErrorConformalPredictor
    xgboost.XGBSklearnAbsoluteErrorConformalPredictor
    xgboost.XGBoosterLeafNodeScaledConformalPredictor
    xgboost.XGBSklearnLeafNodeScaledConformalPredictor
    xgboost.XGBoosterSplitLeafNodeScaledConformalPredictor
            
dispatchers module
------------------

.. autosummary::
    :toctree: api/

    dispatchers.get_absolute_error_conformal_predictor
    dispatchers.get_leaf_node_scaled_conformal_predictor
    dispatchers.get_leaf_node_split_conformal_predictor
             
helpers module
------------------

.. autosummary::
    :toctree: api/

    helpers.gather_intervals
    helpers.check_response_within_interval
    helpers.check_interval_width
    helpers.prepare_prediction_interval_df
    helpers.create_interval_buckets
        