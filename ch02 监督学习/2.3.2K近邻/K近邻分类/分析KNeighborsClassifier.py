from mglearn import discrete_scatter
from mglearn.datasets import make_forge
from mglearn.plot_2d_separator import plot_2d_separator
from sklearn.neighbors import KNeighborsClassifier

from myImport import *

if __name__ == '__main__':
    # 加载数据
    X, y = make_forge()

    # 分析KNeighborsClassifier
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for n_neighbors, ax in zip([1, 3, 9], axes):
        # fit方法返回对象本身，所以我们可以将实例化和拟合放在一行代码中
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
        plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
        discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title("{} neighbor(s)".format(n_neighbors))
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
    axes[0].legend(loc=3)
    plt.savefig("分析KNeighborsClassifier.png")
    plt.show()
