from mglearn.datasets import make_forge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    # 加载数据
    X, y = make_forge()

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)

    # print("X_train shape:{}".format(X_train.shape))  # X_train shape:(19, 2)
    # print("y_train shape:{}".format(y_train.shape))  # y_train shape:(19,)
    # print("X_test shape:{}".format(X_test.shape))  # X_test shape:(7, 2)
    # print("y_test shape:{}".format(y_test.shape))  # y_test shape:(7,)

    # 构建模型
    clf = KNeighborsClassifier(n_neighbors=3)

    # 训练
    clf.fit(X_train, y_train)

    print("test set predictions: {}".format(clf.predict(X_test)))  # test set predictions: [1 1 0 0 0 1 0]
    print("test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))  # test set accuracy: 0.86

