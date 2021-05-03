import numpy as np
import xgboost as xgb
from sklearn.datasets import load_diabetes, load_breast_cancer

import pytest


@pytest.fixture
def np_2x1_with_label():
    """Return 2x1 np array with label."""

    return np.array([[0], [1]]), np.array([[0], [1]])


@pytest.fixture
def np_4x2_with_label():
    """Return 4x2 np array with label."""

    return np.array([[1, 1], [1, 0], [0, 1], [0, 0]]), np.array([90, 10, 0, 0])


@pytest.fixture
def dmatrix_2x1_with_label(np_2x1_with_label):
    """Create an 2x1 xgb.DMatrix with response."""

    xgb_data = xgb.DMatrix(data=np_2x1_with_label[0], label=np_2x1_with_label[1])

    return xgb_data


@pytest.fixture
def dmatrix_2x1_with_label_gamma():
    """Create an 2x1 xgb.DMatrix with (positive) response."""

    xgb_data = xgb.DMatrix(data=np.array([[0], [1]]), label=np.array([[1], [2]]))

    return xgb_data


@pytest.fixture
def dmatrix_4x2_with_label(np_4x2_with_label):
    """Create an 4x2 xgb.DMatrix with response."""

    xgb_data = xgb.DMatrix(data=np_4x2_with_label[0], label=np_4x2_with_label[1])

    return xgb_data


@pytest.fixture
def xgboost_1_split_1_tree(dmatrix_2x1_with_label):
    """Build a dummy xgboost model with a single split on 1 variable."""

    model = xgb.train(
        params={"max_depth": 1}, dtrain=dmatrix_2x1_with_label, num_boost_round=1
    )

    return model


@pytest.fixture
def xgb_regressor_1_split_1_tree(np_2x1_with_label):
    """Build a dummy xgb.XGBRegressor model with a single split on 1 variable."""

    model = xgb.XGBRegressor(max_depth=1, n_estimators=1)

    model.fit(
        X=np_2x1_with_label[0],
        y=np_2x1_with_label[1],
    )

    return model


@pytest.fixture
def xgboost_2_split_1_tree(dmatrix_4x2_with_label):
    """Build a dummy xgboost model with 3 leaves, splitting on 2 variables."""

    model = xgb.train(
        params={
            "objective": "reg:squarederror",
            "max_depth": 2,
            "subsample": 1,
            "colsample_bytree": 1,
            "eta": 1,
            "lambda": 0,
            "gamma": 0,
            "alpha": 0,
        },
        dtrain=dmatrix_4x2_with_label,
        num_boost_round=1,
    )

    return model


@pytest.fixture
def xgboost_2_split_2_tree(dmatrix_4x2_with_label):
    """Build a dummy xgboost model with 4 leaves, splitting on 2 variables across 2 trees."""

    model = xgb.train(
        params={
            "objective": "reg:squarederror",
            "max_depth": 1,
            "subsample": 1,
            "colsample_bytree": 1,
            "eta": 1,
            "lambda": 0,
            "gamma": 0,
            "alpha": 0,
        },
        dtrain=dmatrix_4x2_with_label,
        num_boost_round=2,
    )

    return model


def split_sklearn_data_into_4(data):
    """Function to take a sklearn data bundle and split it into 4 samples."""

    np.random.seed(1)
    random_col = np.random.random(data["data"].shape[0])
    sample_col = np.ones(random_col.shape)
    sample_col[random_col > 0.55] = 2
    sample_col[random_col > 0.7] = 3
    sample_col[random_col > 0.85] = 4

    X_train = data["data"][sample_col == 1]
    X_validate = data["data"][sample_col == 2]
    X_interval = data["data"][sample_col == 3]
    X_test = data["data"][sample_col == 4]

    y_train = data["target"][sample_col == 1]
    y_validate = data["target"][sample_col == 2]
    y_interval = data["target"][sample_col == 3]
    y_test = data["target"][sample_col == 4]

    return (
        X_train,
        y_train,
        X_validate,
        y_validate,
        X_interval,
        y_interval,
        X_test,
        y_test,
    )


def create_4_DMatrices(four_samples_X_y):
    """Function to create 4 xgb.DMatrix objects using consecutive pairs for X and y in each."""

    train = xgb.DMatrix(data=four_samples_X_y[0], label=four_samples_X_y[1])
    validate = xgb.DMatrix(data=four_samples_X_y[2], label=four_samples_X_y[3])
    interval = xgb.DMatrix(data=four_samples_X_y[4], label=four_samples_X_y[5])
    test = xgb.DMatrix(data=four_samples_X_y[6], label=four_samples_X_y[7])

    return train, validate, interval, test


@pytest.fixture(scope="session")
def split_diabetes_data_into_4():
    """Split the diabetes data from sklearn into 4 samples and return np arrays."""

    diabetes = load_diabetes()

    return split_sklearn_data_into_4(diabetes)


@pytest.fixture(scope="session")
def diabetes_xgb_data(split_diabetes_data_into_4):
    """Create 4 xgb.DMatrix objects for the data samples created in split_diabetes_data_into_4."""

    return create_4_DMatrices(split_diabetes_data_into_4)


@pytest.fixture(scope="session")
def split_breast_cancer_data_into_4():
    """Split the breast_cancer data from sklearn into 4 samples and return np arrays."""

    breast_cancer = load_breast_cancer()

    return split_sklearn_data_into_4(breast_cancer)


@pytest.fixture(scope="session")
def breast_cancer_xgb_data(split_breast_cancer_data_into_4):
    """Create 4 xgb.DMatrix objects for the data samples created in split_breast_cancer_data_into_4."""

    return create_4_DMatrices(split_breast_cancer_data_into_4)
