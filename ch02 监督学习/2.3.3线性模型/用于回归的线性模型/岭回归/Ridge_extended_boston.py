from mglearn.datasets import load_extended_boston
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split

from myImport import *

if __name__ == '__main__':
    # 加载数据
    X, y = load_extended_boston()

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # print("X_train shape:{}".format(X_train.shape))  # X_train shape:(75, 1)
    # print("y_train shape:{}".format(y_train.shape))  # y_train shape:(75,)
    # print("X_test shape:{}".format(X_test.shape))  # X_test shape:(25, 1)
    # print("y_test shape:{}".format(y_test.shape))  # y_test shape:(25,)

    # 构建、训练模型
    lr = LinearRegression().fit(X_train, y_train)

    print("lr train set accuracy: {:.2f}".format(lr.score(X_train, y_train)))  # 0.95
    print("lr test set accuracy: {:.2f}".format(lr.score(X_test, y_test)))  # 0.61

    ridge = Ridge().fit(X_train, y_train)

    print("lr train set accuracy: {:.2f}".format(ridge.score(X_train, y_train)))  # 0.89
    print("lr test set accuracy: {:.2f}".format(ridge.score(X_test, y_test)))  # 0.75

    ridge10 = Ridge(alpha=10).fit(X_train, y_train)

    print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))  # 0.79
    print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))  # 0.64

    ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)

    print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))  # 0.93
    print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))  # 0.77

    plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
    plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
    plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
    plt.plot(lr.coef_, 'o', label="LinearRegression")
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.hlines(0, 0, len(lr.coef_))
    plt.ylim(-25, 25)
    plt.legend()
    plt.savefig("Ridge_LinearRegression.png")
    plt.show()
