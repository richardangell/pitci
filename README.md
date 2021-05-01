# Prediction Intervals for Trees using Conformal Intervals - pitci

A package to allow prediction intervals to be generated with tree based models using conformal intervals.

----

The basic idea of (inductive) conformal intervals is to use a calibration set to learn given quantile of the error distribution on that set. This quantile is used as the basis for prediction intervals on new data.

However it is not very useful in default state - which gives the same interval for every new prediction. Instead we want to scale this interval according to the input data. Intuitively we want to increase the interval where we have less confidence about the data and associated prediction and decrease it where we have more confidence.

In order to produce a value that captures the confidence or familiarity we have we some data, compared to our calibration set, pitci uses the number of times each leaf node used to generate a prediction was visited across all rows of the calibration set and then summed across trees.

