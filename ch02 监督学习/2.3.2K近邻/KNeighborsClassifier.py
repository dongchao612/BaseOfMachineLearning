import mglearn
import matplotlib.pyplot as plt
from mglearn.datasets import make_forge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    # mglearn.plots.plot_knn_classification(n_neighbors=3)
    # plt.savefig("plot_knn_classification_3.png")
    # plt.show()

    X, y = make_forge()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # print("X_train shape:{}".format(X_train.shape))  # X_train shape:(19, 2)
    # print("y_train shape:{}".format(y_train.shape))  # y_train shape:(19,)
    # print("X_test shape:{}".format(X_test.shape))  # X_test shape:(7, 2)
    # print("y_test shape:{}".format(y_test.shape))  # y_test shape:(7,)

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    print("test set predictions: {}".format(clf.predict(X_test)))  # test set predictions: [1 1 0 0 0 1 0]
    print("test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))  # test set accuracy: 0.86

    # 分析KNeighborsClassifier
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for n_neighbors, ax in zip([1, 3, 9], axes):
        # fit方法返回对象本身，所以我们可以将实例化和拟合放在一行代码中
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title("{} neighbor(s)".format(n_neighbors))
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
    axes[0].legend(loc=3)
    plt.savefig("KNeighborsClassifier.png")
    plt.show()
