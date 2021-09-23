<p align="center">
  <img src="https://github.com/richardangell/pitci/raw/master/logo.png">
</p>

# Prediction Intervals for Trees using Conformal Intervals

![PyPI](https://img.shields.io/pypi/v/pitci?color=success&style=plastic)
![Build](https://github.com/richardangell/pitci/actions/workflows/python-package.yml/badge.svg?branch=master)
![GitHub](https://img.shields.io/github/license/richardangell/pitci)

## Introduction

The basic idea of inductive conformal intervals is to use a calibration set to learn a given quantile of the error distribution on that set. This quantile is used as the basis for prediction intervals on new data.

However this is often not especially useful in practice as every new prediction will recieve the same interval. Instead we want to scale this interval according to the input data. Intuitively we want to increase the interval where we have less confidence about the data and associated prediction and decrease it where we have more confidence.

In order to produce a scaling factor value captures the confidence or familiarity we have with some data compared to our calibration set, `pitci` uses the number of training data rows that fell into the specific leaf nodes that were visited in making the prediction, summed across all trees. 

For a full list of the supported libraries and more detail on the methods implmeneted, see the [docs](https://pitci.readthedocs.io/en/latest/quick-start.html#external-library-support).

## Install

The easiest way to get `pitci` is directly from [PyPI](https://pypi.org/project/pitci/) using;

```
pip install pitci
```

## Documentation

The documentation for `pitci` can be found [here](https://pitci.readthedocs.io/en/latest/).

For information on how to build the documentation locally see the docs [README](https://github.com/richardangell/pitci/tree/master/docs).

## Examples

There are various example notebooks demonstrating how to use the package in the [examples folder](https://github.com/richardangell/pitci/tree/master/examples) in the repo.


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

