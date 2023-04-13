from mglearn.plot_knn_regression import plot_knn_regression

from myImport import *

if __name__ == '__main__':
    plot_knn_regression(n_neighbors=3)
    plt.savefig("plot_knn_regression_3.png")
    plt.show()