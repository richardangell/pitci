Quick Start
====================

Welcome to the quick start guide for 

.. image:: ../../logo.png

Installation
--------------------

The easiest way to get ``pitci`` is to install directly from ``pip``;

   .. code::

     pip install pitci

Methods Summary
--------------------

``pitci`` allows the user to generate intervals about predictions when using tree based models. 
Conformal intervals are the underlying technqiue that makes this possible. Here we use
*inductive* conformal intervals learn an expected interval width at a given confidence level 
(``alpha``) from a calibration dataset and then this interval is applied to new examples when 
making predictions.

The above is the most basic conformal predictor implemented in ``pitci`` referred to as the
``Absolute Error`` predictors. They use the absolute error between response and 
predictions on the caibration dataset as the set of values from which the baseline interval
will be calculated, by selecting the ``alpha`` quantile. The approach being if we see ``alpha`` % 
of the absolute errors on the calibration sample less than x; then we expect the unseen 
response to fall within plus or minus x of our predictions ``alpha`` % of the time going 
forwards. This approach produces the same interval for every prediction, so may not be 
especially useful in practice.

Instead we often want the prediction intervals to vary based off the input data. ``pitci``
also implements scaled conformal intervals whereby the baseline intervals are shrunk
if we have more confidence in a particular data item or prediction and widened if we have 
less confidence. The challenge is how to calculate these scaling factors that will allow
the intervals to vary depending on the data.

The ``Leaf Node Scaled Absolute Error`` predictors use the total number of data points 
that appeared in the specific leaf nodes used to make a prediction, when the model 
was trained -  as the basis for the scaling factors. Intuitively, we may expect to make 
better predictions for leaf nodes that had more data partitioned into them when building 
the underlying model. In order to make intervals shrink where we have more
confidence the reciprocal of the total leaf node counts is used as the scaling factor
i.e. larger leaf node count means a smaller scaling factor and hence a smaller interval
when multiplying the baseline interval by the scaling factor.

The final type of predictor implemented in ``pitci`` are the ``Split Leaf Node 
Scaled Absolute Error`` predictors. Here the data is split into bins according to the
leaf node count scaling factors and each bin has a different baseline interval that is
calibrated to the desired ``alpha`` level. This can be useful over and above the previous 
predictor type where for a given sample of data the prediction intervals produced as 
well calibrated at the ``alpha`` level however, if the data is subset then these subsets
are no longer well calibrated at ``alpha``.

External Library Support
------------------------------

Currently only certain libraries that implement tree based algorithms are supported by ``pitci``, 
the table below details which techniques are available for which model objects in which libraries;

================= =============== ================================ ======================================
Library           Absolute Error  Leaf Node Scaled Absolute Error  Split Leaf Node Scaled Absolute Error
================= =============== ================================ ======================================
xgboost Booster   x               x                                x
xgboost SKLearn   x               x
lightgbm Booster  x               x                                x
================= =============== ================================ ======================================

The intention is to add support for more libraries in the future!

Only certain objective functions are supported for each ``pitci`` predictor, these are checked 
when initialising the object.

Creating ``pitci`` Predictors
---------------------------------

The easiest way to create the relevant ``pitci`` predictor object for your given model type is 
to use the dispatching functions demonstrated below.

For ``Absolute Error`` conformal predictors;

   .. code::
    
     pitci.get_absolute_error_conformal_predictor(model)

For ``Leaf Node Scaled Absolute Error`` conformal predictors;

   .. code::

     pitci.get_leaf_node_scaled_conformal_predictor(model)

For ``Split Leaf Node Scaled Absolute Error`` conformal predictors;

   .. code::

     pitci.get_split_leaf_node_scaled_conformal_predictor(model)

where ``model`` is the underlying model that the user has built using 
one of the supported libraries.

These functions will return the correct ``pitci`` predictor given the type of ``model``.

If desired, the user can still find the right class from the relevant module (e.g. 
``pitci.xgboost`` or ``pitci.lightgbm``) and initialise it directly.

Calibrate and Predict
---------------------------------

All the predictors implemented in ``pitci`` require calibration before they are ready 
to produce predictions with associated intervals.

This means the user must run the ``calibrate`` method first, after initialising the 
``pitci`` predictor.

When calibrating the user must specify the ``data`` to calibrate on, the ``response`` 
values for the ``data`` and the ``alpha`` level of significance that the intervals 
should be calibrated at;

   .. code::

     pitci_predictor = pitci.get_absolute_error_conformal_predictor(model)
     pitci_predictor.calibrate(
         data=data, response=response, alpha=alpha   
     )

When calibrating a ``pitci`` predictor that uses leaf node scaling, the user has the 
option to pass another dataset in the ``train_data`` argument. If used, this should 
be the same dataset as was used to train the underlying ``model``. This ``train_data`` 
is used to calibrate the leaf node counts, but not the baseline interval width or widths.
The reason for this is that calibration of the leaf node counts does not need to be 
done on a separate sample, like calibrating the baseline intervals. The leaf node count 
visits should ideally come from the training of the model but the underlying modelling 
libraries may not record this information.

   .. code::

     pitci_predictor = pitci.get_leaf_node_scaled_conformal_predictor(model)
     pitci_predictor.calibrate(
         data=data, response=response, alpha=alpha, train_data=train_data 
     )

The ``pitci`` predictors all implement a method called ``predict_with_interval`` which 
produces predictions from the underlying model along with prediction intervals using 
relevent conformal intervals technique.

The only item that the user needs to pass to ``predict_with_interval`` is the ``data`` 
to predict on;

   .. code::

     pitci_predictor.predict_with_interval(data)

The output from ``predict_with_interval`` will be an array of shape n x 3 where the 
first column are the lower prediction intervals, the second column are the predictions 
from the underlying model and the third column are the upper prediction intervals.

Examples
---------------------------------

There are example notebooks available on `Github <https://github.com/richardangell/pitci/tree/master/examples/>`_.
