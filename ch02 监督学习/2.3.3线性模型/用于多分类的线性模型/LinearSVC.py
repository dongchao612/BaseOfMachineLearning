from mglearn import discrete_scatter
from mglearn.plot_2d_separator import plot_2d_classification
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

from myImport import *

if __name__ == '__main__':
    X, y = make_blobs(random_state=42)

    linear_svm = LinearSVC().fit(X, y)
    print("Coefficient shape: ", linear_svm.coef_.shape)
    print("Intercept shape: ", linear_svm.intercept_.shape)


