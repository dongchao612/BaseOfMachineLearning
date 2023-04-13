from mglearn import discrete_scatter
from mglearn.datasets import make_forge
from pandas.plotting import scatter_matrix

from myImport import *

if __name__ == '__main__':
    X, y = make_forge()
    discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend(['Class 0', 'Class 1'], loc=4)
    plt.savefig("forge.png")
    plt.show()

    print("X.shape{}".format(X.shape))  # X.shape(26, 2)
    print({n: v for n, v in zip(['Class 0', 'Class 1'], np.bincount(y))})  # {'Class 0': 13, 'Class 1': 13}

    _dataframe = pd.DataFrame(X, columns=['Class 0', 'Class 1'])
    scatter_matrix(_dataframe, c=y, figsize=(15, 15), marker='o', hist_kwds={'bins': 5},cmap=mglearn.cm3)
    plt.savefig("forge_matrix.png")
    plt.show()
