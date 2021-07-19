# Prediction Intervals for Trees using Conformal Intervals - `pitci`

A package to allow prediction intervals to be generated with tree based models using conformal intervals.

The basic idea of inductive conformal intervals is to use a calibration set to learn given quantile of the error distribution on that set. This quantile is used as the basis for prediction intervals on new data.

However it is not very useful in it's default state - which gives the same interval for every new prediction. Instead we want to scale this interval according to the input data. Intuitively we want to increase the interval where we have less confidence about the data and associated prediction and decrease it where we have more confidence.

In order to produce a value that captures the confidence or familiarity we have with some data compared to our calibration set, `pitci` uses the number of times each leaf node used to generate a particular prediction was visited across all rows of the calibration set and then summed across trees.

## Install

The easiest way to get `pitci` is directly from [PyPI](https://pypi.org/project/pitci/) using;

```
pip install pitci
```

## Build

`pitci` uses [flit](https://flit.readthedocs.io/en/latest/index.html) as the package build tool. 

To install `pitci` for development, use the following commands from the root directory;

```
pip install "flit>=2,<4"
flit install
```

The default `deps` flag for `flit` is `all` so this will install all of the libraries required for testing and creating the docs.

To install `pitci` in editable mode (i.e. the equivalent of `pip install . -e`) use the `symlink` flag;

```
flit install --symlink
```

See the [flit docs](https://flit.readthedocs.io/en/latest/cmdline.html#) for all the command line options for `flit`.


