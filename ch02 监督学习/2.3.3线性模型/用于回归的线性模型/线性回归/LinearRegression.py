from mglearn.datasets import make_wave
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # 加载数据
    X, y = make_wave(n_samples=60)

    # 化分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # print("X_train shape:{}".format(X_train.shape))  # X_train shape:(75, 1)
    # print("y_train shape:{}".format(y_train.shape))  # y_train shape:(75,)
    # print("X_test shape:{}".format(X_test.shape))  # X_test shape:(25, 1)
    # print("y_test shape:{}".format(y_test.shape))  # y_test shape:(25,)

    # 构建模型
    lr = LinearRegression()

    # 训练
    lr.fit(X_train, y_train)

    print("lr.corf_: {}".format(lr.coef_))  # lr.corf_: [0.39390555]
    print("lr.intercept_:{}".format(lr.intercept_))  # lr.intercept_:-0.031804343026759746
    print("lr train set accuracy: {:.2f}".format(lr.score(X_train, y_train)))  # 0.67
    print("lr test set accuracy: {:.2f}".format(lr.score(X_test, y_test)))  # 0.66
