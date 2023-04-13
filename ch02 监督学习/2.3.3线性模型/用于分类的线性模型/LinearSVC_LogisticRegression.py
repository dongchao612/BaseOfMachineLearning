from mglearn.datasets import make_forge
from mglearn.plot_2d_separator import plot_2d_separator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from myImport import *

if __name__ == '__main__':

    X, y = make_forge()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
        clf = model.fit(X, y)
        plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title("{}".format(clf.__class__.__name__))
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
    axes[0].legend()
    plt.savefig("LinearSVC_LogisticRegression.png")
    plt.show()
