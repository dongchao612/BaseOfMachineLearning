from mglearn import discrete_scatter
from sklearn.datasets import make_blobs

from myImport import *

if __name__ == '__main__':
    X, y = make_blobs(random_state=42)
    discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend(["Class 0", "Class 1", "Class 2"])
    plt.savefig("plot_linear_svc.png")
    plt.show()