from mysklearn import *

if __name__ == '__main__':
    # mglearn.plots.plot_linear_regression_wave()
    # plt.show()

    X, y = make_wave()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("X_train shape:{}".format(X_train.shape))  # X_train shape:(75, 1)
    print("y_train shape:{}".format(y_train.shape))  # y_train shape:(75,)

    print("X_test shape:{}".format(X_test.shape))  # X_test shape:(25, 1)
    print("y_test shape:{}".format(y_test.shape))  # y_test shape:(25,)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print("lr.corf_: {}".format(lr.coef_))  # lr.corf_: [0.46196473]
    print("lr.intercept_:{}".format(lr.intercept_))  # lr.intercept_:-0.001525054558320263
    print("lr train set accuracy: {:.2f}".format(lr.score(X_train, y_train)))
    print("lr test set accuracy: {:.2f}".format(lr.score(X_test, y_test)))

    X, y = load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("X_train shape:{}".format(X_train.shape))  # X_train shape:(379, 104)
    print("y_train shape:{}".format(y_train.shape))  # y_train shape:(379,)

    print("X_test shape:{}".format(X_test.shape))  # X_test shape:(127, 104)
    print("y_test shape:{}".format(y_test.shape))  # y_test shape:(127,)
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print("lr.corf_: {}".format(lr.coef_))  # lr.corf_: (104,)
    print("lr.intercept_:{}".format(lr.intercept_))
    print("lr train set accuracy: {:.2f}".format(lr.score(X_train, y_train)))
    print("lr test set accuracy: {:.2f}".format(lr.score(X_test, y_test)))

    ridge = Ridge()
    ridge10 = Ridge(alpha=10)
    ridge01 = Ridge(alpha=0.1)

    ridge.fit(X_train, y_train)
    ridge10.fit(X_train, y_train)
    ridge01.fit(X_train, y_train)

    print("ridge train set accuracy: {:.2f}".format(ridge.score(X_train, y_train)))
    print("ridge test set accuracy: {:.2f}".format(ridge.score(X_test, y_test)))

    print("ridge10 train set accuracy: {:.2f}".format(ridge10.score(X_train, y_train)))
    print("ridge10 test set accuracy: {:.2f}".format(ridge10.score(X_test, y_test)))

    print("ridge01 train set accuracy: {:.2f}".format(ridge01.score(X_train, y_train)))
    print("ridge01 test set accuracy: {:.2f}".format(ridge01.score(X_test, y_test)))

    # plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
    # plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
    # plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
    # plt.plot(lr.coef_, 'o', label="LinearRegression")
    # plt.xlabel("Coefficient index")
    # plt.ylabel("Coefficient magnitude")
    # plt.hlines(0, 0, len(lr.coef_))
    # plt.ylim(-25, 25)
    # plt.legend()
    # plt.savefig("ridge.png")
    # plt.show()

    # mglearn.plots.plot_ridge_n_samples()
    # plt.show()

    lasso = Lasso()
    lasso.fit(X_train, y_train)
    print("lasso train set accuracy: {:.2f}".format(lasso.score(X_train, y_train)))
    print("lasso test set accuracy: {:.2f}".format(lasso.score(X_test, y_test)))
    print("lasso number of features used:{}".format(np.sum(lasso.coef_ != 0)))

    lasso001 = Lasso(alpha=0.01, max_iter=100000)
    lasso001.fit(X_train, y_train)
    print("lasso001 train set accuracy: {:.2f}".format(lasso001.score(X_train, y_train)))
    print("lasso001 test set accuracy: {:.2f}".format(lasso001.score(X_test, y_test)))
    print("lasso001 number of features used:{}".format(np.sum(lasso001.coef_ != 0)))

    lasso00001 = Lasso(alpha=0.0001, max_iter=100000)
    lasso00001.fit(X_train, y_train)
    print("lasso00001 train set accuracy: {:.2f}".format(lasso00001.score(X_train, y_train)))
    print("lasso00001 test set accuracy: {:.2f}".format(lasso00001.score(X_test, y_test)))
    print("lasso00001 number of features used:{}".format(np.sum(lasso00001.coef_ != 0)))

    plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
    plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.1")
    plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.00001")
    plt.plot(ridge01.coef_, 'o', label="Ridge alpha=1")
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.hlines(0, 0, len(lr.coef_))
    plt.ylim(-25, 25)
    plt.legend()
    plt.savefig("lasso.png")
    plt.show()
