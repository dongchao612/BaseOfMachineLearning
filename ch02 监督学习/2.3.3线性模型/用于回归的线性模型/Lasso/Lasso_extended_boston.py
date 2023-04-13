from mglearn.datasets import load_extended_boston
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split

from myImport import *

if __name__ == '__main__':
    # 加载数据
    X, y = load_extended_boston()

    # 化分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # 构建、训练模型

    ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)

    print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))# 0.93
    print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))# 0.77

    lasso = Lasso().fit(X_train, y_train)

    print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))# 0.29
    print("Test set score: {:.2f}".format(lasso.score(X_test, y_test))) # 0.29
    print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))# 4

    lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)

    print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
    print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))

    lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)

    print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
    print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))

    plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
    plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
    plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
    plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
    plt.legend(ncol=2, loc=(0, 1.05))
    plt.ylim(-25, 25)
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.savefig("Lasso_Ridge_.png")
    plt.show()