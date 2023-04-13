from mglearn.datasets import make_wave
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

if __name__ == '__main__':
    # 加载数据
    X, y = make_wave(n_samples=40)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # print("X_train shape:{}".format(X_train.shape))  # X_train shape:(75, 1)
    # print("y_train shape:{}".format(y_train.shape))  # y_train shape:(75,)
    # print("X_test shape:{}".format(X_test.shape))  # X_test shape:(25, 1)
    # print("y_test shape:{}".format(y_test.shape))  # y_test shape:(25,)

    reg = KNeighborsRegressor(n_neighbors=3)
    reg.fit(X_train, y_train)

    print("test set predictions: \n{}".format(reg.predict(X_test)))
    print("test set R^2: {:.2f}".format(reg.score(X_test, y_test)))  # test set R^2: 0.83
