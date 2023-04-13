from mglearn.plot_knn_classification import plot_knn_classification

from myImport import *

if __name__ == '__main__':
    plot_knn_classification(n_neighbors=1)
    plt.savefig("plot_knn_classification_1.png")
    plt.show()