# lib
import sys
import IPython
import mglearn
import sklearn
import matplotlib

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# datasets
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer

from mglearn.datasets import load_extended_boston
from mglearn.datasets import make_forge
from mglearn.datasets import make_wave

# model_selection
from sklearn.model_selection import train_test_split

# --------------K近邻-----------------
# neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

# plot_knn_regression
from mglearn.plot_knn_regression import plot_knn_regression
from mglearn.plot_knn_classification import plot_knn_classification

# --------------线性模型-----------------
# linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

# plot_linear_regression_wave
from mglearn.plot_linear_regression import plot_linear_regression_wave
from mglearn.plot_ridge import plot_ridge_n_samples
